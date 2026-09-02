from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    NVFP4,
    QuantizationScheme,
)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

init_dist()
MODEL_ID = "tencent/Hy3"

with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={},
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
)
ds = ds.shuffle(seed=42)


def preprocess(example):
    BOS = "<｜begin▁of▁sentence｜>"
    EOS = "<｜end▁of▁sentence｜>"
    text = BOS
    for message in example["messages"]:
        role = message["role"]
        content = message["content"]
        if role == "system":
            text += content
        elif role == "user":
            text += f"<｜User｜>{content}"
        elif role == "assistant":
            text += f"<｜Assistant｜></think>{content}{EOS}"
    return {"text": text}


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = [
    REAPPruningModifier(sparsity=0.25),
    QuantizationModifier(
        config_groups={
            "experts": QuantizationScheme(
                targets=[
                    r"re:.*mlp.*$",
                ],
                **NVFP4,
            ),
        },
        ignore=[
            "re:.*lm_head",
            "re:.*mlp.gate$",
        ],
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    batch_size=1,
    shuffle_calibration_samples=True,
    propagate_error=False,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-REAP-25"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
