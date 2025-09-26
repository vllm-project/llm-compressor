from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
SAVE_DIR = MODEL_ID.split("/")[-1] + "-W4A16-awq"

# Configure the quantization algorithm to run.
recipe = [
    AWQModifier(
        duo_scaling=False,
        ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        scheme="W4A16",
        targets=["Linear"],
    ),
]

# Select calibration dataset.
DATASET_ID = "codeparrot/self-instruct-starcoder"
DATASET_SPLIT = "curated"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048


def get_calib_dataset(tokenizer):
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*10}]",
    )

    def preprocess(example):
        chat_messages = [
            {"role": "user", "content": example["instruction"].strip()},
            {"role": "assistant", "content": example["output"].strip()},
        ]
        tokenized_messages = tokenizer.apply_chat_template(chat_messages, tokenize=True)
        return {"input_ids": tokenized_messages}

    ds = (
        ds.shuffle(seed=42)
        .map(preprocess, remove_columns=ds.column_names)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )
    return ds


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ###
    ### Apply algorithms.
    ###
    oneshot(
        model=model,
        dataset=get_calib_dataset(tokenizer),
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        log_dir=None,
    )

    # Confirm generations of the quantized model look sane.
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    input_ids = tokenizer(
        "Write a binary search function", return_tensors="pt"
    ).input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=150)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    # Save model to disk
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
