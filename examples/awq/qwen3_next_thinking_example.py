import torch
from compressed_tensors.offload import dispatch_model
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# NOTE: Use this example when deploying Qwen3-Next in thinking mode.
# For non-thinking deployment, use qwen3_next_example.py instead.
# Using a reasoning dataset with loss_mask ensures AWQ weight scales
# are fit to the thinking-mode activation distribution.

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Thinking"
SAVE_DIR = MODEL_ID.split("/")[-1] + "-W4A16-awq-thinking"

DATASET_ID = "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 4096

recipe = [
    AWQModifier(
        ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        scheme="W4A16",
        targets=["Linear"],
    ),
]


def get_calib_dataset(tokenizer):
    raw_ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES * 5}]",
    ).shuffle(seed=42)

    rows = []
    for example in raw_ds:
        if len(rows) >= NUM_CALIBRATION_SAMPLES:
            break

        messages = [
            {"role": "user",      "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        tokens = tokenizer(
            tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=True),
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )
        seq_len = len(tokens["input_ids"])

        # Build loss_mask: 1 for assistant tokens, 0 for prompt tokens.
        prompt_ids = tokenizer(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": example["instruction"]}],
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = min(len(prompt_ids), seq_len)

        rows.append({
            "input_ids":      tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "loss_mask":      [0] * prompt_len + [1] * (seq_len - prompt_len),
        })

    return Dataset.from_list(rows)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(val).unsqueeze(0) for key, val in batch[0].items()}


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    oneshot(
        model=model,
        dataset=get_calib_dataset(tokenizer),
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        data_collator=data_collator,
        use_loss_mask=True,
    )

    print("========== SAMPLE GENERATION (thinking mode) ==============")
    dispatch_model(model)
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 17 * 24? Think step by step."}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(input_ids, max_new_tokens=512)
    print(tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=False))
    print("==========================================================\n")

    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
