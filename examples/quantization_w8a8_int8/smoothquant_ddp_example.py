"""
Distributed SmoothQuant + GPTQ W8A8 quantization using Data-Parallel Calibration.

Run with:
    torchrun --standalone --nproc_per_node=NUM_GPUS smoothquant_ddp_example.py

Each rank loads a disjoint partition of the calibration dataset.
SmoothQuantModifier all-reduces per-channel activation statistics across ranks
before computing smoothing scales (identical on every rank, no weight broadcast
needed). GPTQModifier then applies distributed W8A8 quantization.

This script intentionally mirrors the structure of
examples/quantization_w4a16/llama3_ddp_example.py so it is easy to diff.
"""

import time

import torch
import torch.distributed as dist
from compressed_tensors.offload import dispatch_model, init_dist, load_offloaded_model
from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-7B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# ---------------------------------------------------------------------------
# DDP init + model load
# ---------------------------------------------------------------------------
init_dist()

with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto_offload",
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ---------------------------------------------------------------------------
# Dataset: each rank gets a disjoint slice
# ---------------------------------------------------------------------------
ds = load_dataset(
    DATASET_ID,
    split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES),
)
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(preprocess)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# ---------------------------------------------------------------------------
# Recipe: SmoothQuant (distributed-aware) + GPTQ W8A8
# ---------------------------------------------------------------------------
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# ---------------------------------------------------------------------------
# Run oneshot
# ---------------------------------------------------------------------------
torch.cuda.reset_peak_memory_stats()
start_time = time.time()

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

elapsed = time.time() - start_time
peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

rank = dist.get_rank()
logger.info(
    f"[Rank {rank}] Done in {elapsed:.1f}s | Peak GPU mem: {peak_mem_gb:.2f} GB"
)

# ---------------------------------------------------------------------------
# Sample generation (rank 0 only)
# ---------------------------------------------------------------------------
# Sample generation (all ranks must participate)
dist.barrier()
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {k: v.to(model.device) for k, v in sample.items()}
output = model.generate(**sample, max_new_tokens=50)
if rank == 0:
    logger.info("\n========== SAMPLE GENERATION ==========")
    logger.info(tokenizer.decode(output[0]))
    logger.info("========================================\n")

# ---------------------------------------------------------------------------
# Save (rank 0 only — save_pretrained handles dist internally)
# ---------------------------------------------------------------------------
if rank == 0:
    SAVE_DIR = (
        MODEL_ID.rstrip("/").split("/")[-1]
        + "-W8A8-SmoothQuant-DDP"
        + str(dist.get_world_size())
    )
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    logger.info(f"[Rank {rank}] Saved to {SAVE_DIR}")

dist.destroy_process_group()
