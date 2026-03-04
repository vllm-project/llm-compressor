from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.qwen3_5_moe import (  # noqa: F401
    CalibrationQwen3_5MoeSparseMoeBlock,
)
from llmcompressor.modifiers.awq import AWQModifier

# Load the model (Qwen3.5 MoE is multimodal, requires AutoModel)
model_id = "Qwen/Qwen3.5-35B-A3B"
model = AutoModel.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# MoE calibration is now handled automatically by the pipeline.
# The `CalibrationQwen3_5MoeSparseMoeBlock` modules
# (from `llmcompressor.modeling.qwen3_5_moe`) will be applied during
# calibration to enable proper expert calibration. These replace the
# original `Qwen3_5MoeSparseMoeBlock` class from
# `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`.

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
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

moe_ignores = [
    # Vision encoder and merger components (quantize text transformer only)
    "model.visual.*",
    "model.merger.*",
    # MoE router gates (not quantizable in vLLM)
    "re:.*mlp\\.gate$",
    "re:.*shared_expert_gate$",
    # Output head
    "lm_head",
]

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with AWQ with a group size 128
recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=moe_ignores)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    calibrate_moe_context=True,
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
