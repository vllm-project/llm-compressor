"""
HOPE Expert Pruning Example with Token Masking (Moonlight)

This is the masked variant of HOPE expert pruning. Each sample has its own
`loss_mask`. Only assistant-response tokens contribute to HOPE's pairwise
expert interaction statistics. So the pruning decision ignores user
prompts and role headers.
"""

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import HOPEPruningModifier
from llmcompressor.utils import load_context

# Note: the Moonlight tokenizer requires the `tiktoken` package and
# trust_remote_code=True.
model_id = "moonshotai/Moonlight-16B-A3B-Instruct"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Special tokens marking assistant response boundaries in the chat format:
#   <|im_assistant|>assistant<|im_middle|>[content]<|im_end|>
im_assistant_id = tokenizer.convert_tokens_to_ids("<|im_assistant|>")
im_middle_id = tokenizer.convert_tokens_to_ids("<|im_middle|>")
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

# open-perfectblend: the dataset behind the "perfectblend" shortcut in
# reap_moonlight_16b.py.
DATASET_ID = "mlabonne/open-perfectblend"
DATASET_SPLIT = "train"
ROLE_MAP = {"human": "user", "gpt": "assistant"}

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess(example):
    messages = [
        {"role": ROLE_MAP.get(msg["from"], msg["from"]), "content": msg["value"]}
        for msg in example["conversations"]
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


ds = ds.map(preprocess)


def tokenize(sample):
    """Tokenize and build loss_mask (1 = assistant-response token)."""
    tokenized = tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

    # mark assistant-response tokens; unlike Llama-3, no BOS token to keep
    input_ids = tokenized["input_ids"]
    mask = [0] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i] == im_assistant_id:
            i += 1
            while i < len(input_ids) and input_ids[i] != im_middle_id:
                i += 1  # skip the literal "assistant" role word
            i += 1  # skip <|im_middle|>
            while i < len(input_ids) and input_ids[i] != im_end_id:
                mask[i] = 1
                i += 1
        else:
            i += 1

    tokenized["loss_mask"] = torch.tensor(mask)
    return tokenized


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Prune 25% of the experts in each MoE layer (64 -> 48 experts). HOPE picks
# the 16 experts whose joint removal takes the least interaction mass out of
# the layer, instead of the 16 with the lowest individual saliency.
recipe = HOPEPruningModifier(
    sparsity=0.25, report_path="moonlight_hope_report_masked.json"
)

# use_loss_mask=True restricts the interaction statistics to loss_mask=1
# tokens: masked-out tokens contribute to neither the pairwise products nor
# the co-selection counts.
oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    use_loss_mask=True,
    moe_calibrate_all_experts=False,  # Disable calibrating all experts for HOPE
)

# Confirm generations of the compressed model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0].tolist()))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-HOPE-25-masked"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
