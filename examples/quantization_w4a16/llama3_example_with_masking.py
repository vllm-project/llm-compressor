"""
GPTQ Quantization Example with Token Masking

This example demonstrates GPTQ quantization with token masking, so that only
assistant responses (plus the BOS token, see below) contribute to the GPTQ
Hessian. User prompts and the chat template's role headers are excluded.

The calibration data is the same as in `llama3_example.py`; only the loss_mask
field and `use_loss_mask=True` are added.
"""

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Get special token IDs for masking logic.
# These are used to identify assistant response boundaries in the chat format.
start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
assistant_id = tokenizer.convert_tokens_to_ids("assistant")

# Select calibration dataset.
# open-perfectblend is a multi-turn conversation dataset (the dataset behind
# the "perfectblend" shortcut used by llama3_example.py).
DATASET_ID = "mlabonne/open-perfectblend"
DATASET_SPLIT = "train"
ROLE_MAP = {"human": "user", "gpt": "assistant"}

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess(example):
    messages = [
        {"role": ROLE_MAP.get(msg["from"], msg["from"]), "content": msg["value"]}
        for msg in example["conversations"]
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


ds = ds.map(preprocess)


def tokenize(sample):
    """
    Tokenize inputs and create loss_mask for assistant responses.

    The mask marks tokens that are part of assistant responses and the BOS token
    with 1, and all other tokens (user messages, role headers) with 0.
    This allows GPTQ to build its Hessian from the assistant outputs only.

    Llama-3 chat format:
        <|start_header_id|>assistant<|end_header_id|>\n\n[content]<|eot_id|>
    """
    # the chat template already contains the BOS token
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

    # Keep the BOS token. Llama-3 uses the first token as an attention sink with
    # very large activations; if it is excluded from the Hessian, GPTQ no longer
    # compensates the quantization error on those activations and accuracy drops
    # sharply (gsm8k 0.74 -> 0.69 for W4A16, 0.61 -> 0.37 for W3A16).
    mask[0] = 1

    tokenized["loss_mask"] = torch.tensor(mask)
    return tokenized


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply algorithms with token masking enabled.
# use_loss_mask=True tells GPTQ to use the loss_mask field from the dataset
# when accumulating the Hessian.
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
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128-masked"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
