import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

# Initialize distributed.
# Usage: torchrun --nproc_per_node=2 llama3_8b_w8a8_distributed.py
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)

if rank == 0:
    print(f"Running distributed quantization with {world_size} GPUs")

# Load model to CPU for sequential onloading.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map=None,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load and partition dataset across ranks.
# Each rank loads a disjoint slice of the calibration data.
samples_per_rank = NUM_CALIBRATION_SAMPLES // world_size
start = samples_per_rank * rank
end = start + samples_per_rank

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[{start}:{end}]")
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
# QuantizationModifier automatically detects torch.distributed and:
#   * partitions weight calibration across ranks
#   * all-reduces activation observer statistics at layer boundaries
recipe = [
    QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=samples_per_rank,
)

# Save to disk compressed (rank 0 only).
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W8A8-distributed"
if rank == 0:
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}")

dist.destroy_process_group()
