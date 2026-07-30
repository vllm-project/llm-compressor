import torch
from compressed_tensors.distributed import init_dist
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.modifiers.pruning import REAPPruningModifier

MODEL_ID = "inference-optimization/Kimi-K3-0.40B"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

init_dist()
model = KimiK3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]",
).shuffle(seed=42)
dataset = dataset.map(
    lambda sample: {
        "text": tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
        )
    }
)
dataset = dataset.map(
    lambda sample: tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    ),
    remove_columns=dataset.column_names,
)

oneshot(
    model=model,
    processor=tokenizer,
    dataset=dataset,
    recipe=REAPPruningModifier(sparsity=0.25),
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=False,
)

save_dir = MODEL_ID.rsplit("/", 1)[-1] + "-REAP-25"
model.save_pretrained(save_dir, save_compressed=True)
tokenizer.save_pretrained(save_dir)
torch.distributed.destroy_process_group()
