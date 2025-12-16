from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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
        # offload_device=torch.device("cpu")
    ),
]

# Select calibration dataset.
DATASET_ID = "codeparrot/self-instruct-starcoder"
DATASET_SPLIT = "curated"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048


def get_calib_dataset(tokenizer):
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES * 10}]",
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
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    model.model.config.num_hidden_layers = 1

    ###
    ### Apply algorithms.
    ###
    import time
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    oneshot(
        model=model,
        dataset=get_calib_dataset(tokenizer),
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        log_dir=None,
    )

    elapsed_time = time.time() - start_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print(f"\n{'='*60}")
    print(f"AWQ Quantization Complete")
    print(f"{'='*60}")
    print(f"Time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
    print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")
    print(f"{'='*60}\n")
