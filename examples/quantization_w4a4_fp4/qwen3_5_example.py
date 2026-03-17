from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE: This example requires transformers >= v5

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"

# Load model.
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoTokenizer.from_pretrained(MODEL_ID)


recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 4096

# Load datasets and preprocess.
samples_per_dataset = NUM_CALIBRATION_SAMPLES

ds_ultrachat = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=f"train_sft[:{samples_per_dataset}]",
)

# Both datasets share a "messages" column with the same chat format.
# Keep only that column so we can concatenate them.
ds = ds_ultrachat.select_columns(["messages"])
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": processor.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return processor(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
