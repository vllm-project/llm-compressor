from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import GPTQModifier

# Select model and load it.

# This script takes about 48 hours on 1xA100 to complete.
# Future improvements will reduce this runtime (#1561, #1558).

# For DeepSeek-R1, we require a full precision model in order to properly calibrate
# `DeepSeek-R1-0528-BF16` is a DeepSeek-V3 FP8 model which has been converted to BF16

model_id = "unsloth/DeepSeek-R1-0528-BF16"
config = AutoConfig.from_pretrained(model_id)
del config.quantization_config  # fp8 qconfig no longer appplies to bf16 model
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", config=config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = replace_modules_for_calibration(model)

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

# Configure the quantization algorithm to run.
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision
recipe = GPTQModifier(
    targets="Linear", scheme="W4A16", ignore=["lm_head", "re:.*mlp.gate$"]
)

# Apply algorithms.
# due to the large size of DeepSeekV3, we specify sequential targets such that
# only one MLP is loaded into GPU memory at a time
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["DeepseekV3Attention", "DeepseekV3MLP"],
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
