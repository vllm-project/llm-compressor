"""
AWQ Quantization Example with Token Masking

This example demonstrates AWQ quantization with token masking to focus the
optimization on assistant responses only. The loss_mask feature allows AWQ
to compute loss only on tokens that matter (assistant outputs), while ignoring
user prompts and special tokens during calibration.

This is particularly useful for instruction-tuned models where you want the
quantization to preserve the quality of generated responses.
"""

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Get special token IDs for masking logic.
# These are used to identify assistant response boundaries in the chat format.
start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
assistant_id = tokenizer.convert_tokens_to_ids("assistant")

# Select calibration dataset.
# ultrachat_200k is a multi-turn conversation dataset with user/assistant messages.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

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


def tokenize(sample):
    """
    Tokenize inputs and create loss_mask for assistant responses.

    The mask marks tokens that are part of assistant responses with 1,
    and all other tokens (user messages, special tokens) with 0.
    This allows AWQ to focus its optimization on the assistant outputs.

    Llama-3 chat format:
        <|start_header_id|>assistant<|end_header_id|>\n\n[content]<|eot_id|>
    """
    tokenized = tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

    # Create mask: 1 for tokens in assistant responses, 0 otherwise
    input_ids = tokenized["input_ids"]
    mask = [0] * len(input_ids)

    # Find all assistant response segments
    i = 0
    while i < len(input_ids) - 2:
        # Look for: <|start_header_id|> assistant <|end_header_id|>
        if (
            input_ids[i] == start_header_id
            and input_ids[i + 1] == assistant_id
            and input_ids[i + 2] == end_header_id
        ):
            i += 3  # Skip header tokens
            # Mark content tokens until eot_id
            while i < len(input_ids) and input_ids[i] != eot_id:
                mask[i] = 1
                i += 1
        else:
            i += 1

    tokenized["loss_mask"] = torch.tensor(mask)
    return tokenized


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
recipe = [
    AWQModifier(
        ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"], duo_scaling="both"
    ),
]

# Apply algorithms with token masking enabled.
# use_loss_mask=True tells AWQ to use the loss_mask field from the dataset
# to focus optimization on assistant responses only.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    use_loss_mask=True,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-asym-masked"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
