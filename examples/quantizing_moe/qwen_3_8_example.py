from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Select model and load it.
init_dist()
model_id = "inference-optimization/Qwen3.8-1.0B-A0.6B"  # Qwen/Qwen3.8-2.4T-A95B
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",
        max_memory={},
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
DATASET_ID = "mlabonne/open-perfectblend"
DATASET_SPLIT = "train"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
)
ds = ds.shuffle(seed=42)


ROLE_MAP = {"human": "user", "gpt": "assistant"}


def preprocess(example):
    messages = [
        {"role": ROLE_MAP.get(msg["from"], msg["from"]), "content": msg["value"]}
        for msg in example["conversations"]
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Create recipe
recipe = [
    QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\..*",
                    r"re:.*linear_attn.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$",
                ],
                **FP8_BLOCK,
            ),
            "mlp": QuantizationScheme(
                targets=[r"re:.*mlp\..*"],
                **NVFP4,
            ),
        },
        ignore=[
            "re:.*lm_head",
            "re:.*mlp.gate$",
            "re:.*shared_expert_gate.*",
        ],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    pipeline="sequential",
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
