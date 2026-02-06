from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.minimax_m2 import (  # noqa: F401
    CalibrationMiniMaxM2SparseMoeBlock,
)
from llmcompressor.modifiers.awq import AWQMapping, AWQModifier

# Load the model
model_id = "MiniMaxAI/MiniMax-M2"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# MoE calibration is handled automatically by the pipeline.
# The `CalibrationMiniMaxM2SparseMoeBlock` modules (from
# `llmcompressor.modeling.minimax_m2`) will be applied during calibration to enable
# proper expert calibration. These replace the original
# `MiniMaxM2SparseMoeBlock` class from
# `transformers.models.minimax_m2.modeling_minimax_m2`.

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
    # MoE gate layers are sensitive to quantization.
    "re:.*mlp.gate$",
    # Ignore the output head.
    "lm_head",
]

# Configure the quantization algorithm to run.
recipe = AWQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=moe_ignores,
    mappings=[
        AWQMapping(
            "re:.*input_layernorm$",
            ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
        )
    ],
)


# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["MiniMaxM2DecoderLayer"],
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
