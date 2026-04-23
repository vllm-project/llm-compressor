"""
Benchmark: Single-GPU vs Multi-GPU DDP SmoothQuant calibration time.

Usage:
    # 1 GPU
    python benchmark_smoothquant_ddp.py --num_gpus 1

    # 2 GPU
    torchrun --standalone --nproc_per_node=2 benchmark_smoothquant_ddp.py --num_gpus 2

    # 4 GPU
    torchrun --standalone --nproc_per_node=4 benchmark_smoothquant_ddp.py --num_gpus 4
"""

import argparse
import time

import torch
import torch.distributed as dist
from compressed_tensors.offload import init_dist, load_offloaded_model
from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
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


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def main(num_gpus: int):
    is_distributed = num_gpus > 1

    # ------------------------------------------------------------------
    # Init distributed if needed
    # ------------------------------------------------------------------
    if is_distributed:
        init_dist()
        with load_offloaded_model():
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype="auto",
                device_map="auto_offload",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype="auto",
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ------------------------------------------------------------------
    # Dataset — each rank gets a disjoint slice
    # ------------------------------------------------------------------
    rank = get_rank()
    world_size = get_world_size()

    samples_per_rank = NUM_CALIBRATION_SAMPLES // world_size
    start = samples_per_rank * rank
    split = f"{DATASET_SPLIT}[{start}:{start + samples_per_rank}]"

    ds = load_dataset(DATASET_ID, split=split)
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

    # ------------------------------------------------------------------
    # Recipe
    # ------------------------------------------------------------------
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
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

    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"BENCHMARK RESULTS — {world_size} GPU(s)")
        logger.info("=" * 60)
        logger.info(f"Model:           {MODEL_ID}")
        logger.info(f"Calibration:     {NUM_CALIBRATION_SAMPLES} samples total")
        logger.info(f"Samples/rank:    {samples_per_rank}")
        logger.info(f"World size:      {world_size}")
        logger.info(f"Total time:      {elapsed:.1f}s ({elapsed/60:.2f} min)")
        logger.info(f"Peak GPU mem:    {peak_mem_gb:.2f} GB (rank 0)")
        logger.info("=" * 60)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()
    main(args.num_gpus)
