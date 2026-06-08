"""
Multi-GPU per group DDP example with AutoRound quantization.

Each rank gets a local GPU group for block-level model parallelism, while
gradients are synchronized across ranks via all_reduce for identical
convergence despite split calibration data.

Usage (4 GPUs, 2 GPUs per group):
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=2 \\
      ddp_qwen3_multi_gpu_example.py \\
      --model /storage/yiliu7/Qwen/Qwen3-8B \\
      --scheme W4A16 \\
      --nsamples 32 --iters 50

For single-GPU DDP:
  torchrun --nproc_per_node=4 ddp_qwen3_multi_gpu_example.py ...
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from compressed_tensors.offload import dispatch_model, load_offloaded_model
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot


def fix_everything(seed=42):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_deterministic():
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    fix_everything()


def init_dist_multi_gpu(gpus_per_group=2):
    """Initialize distributed with multiple GPUs per group.

    ``CUDA_VISIBLE_DEVICES`` must already be set to a disjoint subset of
    GPUs for this rank (the ``launch_multi_gpu.sh`` wrapper handles this).
    NCCL communication uses the first visible GPU (local cuda:0).

    Example with 4 physical GPUs, 2 per group:
      - Rank 0 -> local cuda:0, cuda:1 (physical 0, 1)
      - Rank 1 -> local cuda:0, cuda:1 (physical 2, 3)
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size < 2:
        logger.info("Single-process mode, skipping distributed init")
        return

    # NCCL uses the first visible GPU
    torch.cuda.set_device(0)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda:0"),
    )
    dist.barrier()
    actual_count = torch.cuda.device_count()
    logger.info(
        f"[Rank {rank}/{world_size}] CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
        f"(visible GPUs: {actual_count})"
    )
    if actual_count < gpus_per_group:
        logger.warning(
            f"[Rank {rank}] Expected {gpus_per_group} GPUs but only "
            f"{actual_count} are visible"
        )


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoRound Quantization with Multi-GPU per Group DDP"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name or path",
    )
    parser.add_argument(
        "--gpus-per-group",
        type=int,
        default=2,
        help="Number of GPUs per rank-local group for block sharding (default: 2)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="W4A16",
        help="Quantization scheme (W4A16, MXFP8, MXFP4, etc.)",
    )
    parser.add_argument("--iters", type=int, default=200, help="Number of iterations")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of samples")
    parser.add_argument(
        "--disable_torch_compile",
        action="store_true",
        help="Disable torch.compile for model acceleration during quantization",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for reproducibility",
    )
    args = parser.parse_args()

    if args.deterministic:
        config_deterministic()

    model_id = args.model

    ###### MULTI-GPU DDP INIT #####
    init_dist_multi_gpu(gpus_per_group=args.gpus_per_group)
    # For multi-GPU-per-group AutoRound, keep the base model anchored on the
    # rank-local primary GPU and let AutoRound auto-dispatch each block within
    # the local GPU group during tuning. Pre-sharding the loaded model across
    # the group can leave residual modules and cached activations on different
    # local GPUs before AutoRound takes over.
    load_device_map = "auto"
    if args.gpus_per_group > 1:
        load_device_map = {"": torch.device("cuda:0")}
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype="auto", device_map=load_device_map
        )
    ###############################

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    NUM_CALIBRATION_SAMPLES = args.nsamples
    MAX_SEQUENCE_LENGTH = 2048
    ITERS = args.iters

    # Get aligned calibration dataset.
    from auto_round.calib_dataset import get_dataset  # noqa: E402

    # Note: Make sure model are loaded before importing auto-round related code.
    from llmcompressor.modifiers.autoround import AutoRoundModifier  # noqa: E402

    ds = get_dataset(
        tokenizer=tokenizer,
        seqlen=MAX_SEQUENCE_LENGTH,
        nsamples=NUM_CALIBRATION_SAMPLES,
    )

    # Configure the quantization algorithm.
    recipe = AutoRoundModifier(
        targets="Linear",
        scheme=args.scheme,
        ignore=[
            "lm_head",
            "re:.*mlp.gate$",
        ],
        iters=ITERS,
        enable_torch_compile=not args.disable_torch_compile,
    )

    # Apply algorithms.
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        shuffle_calibration_samples=False,
    )

    rank, world_size = get_dist_info()
    logger.info(f"[Rank {rank}] Quantization completed")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if rank != 0:
        sys.exit(0)

    if rank == 0:
        # Confirm generations of the quantized model look sane.
        logger.info("\n\n")
        logger.info("========== SAMPLE GENERATION ==============")
        dispatch_model(model)
        sample = tokenizer("Hello my name is", return_tensors="pt")
        sample_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sample = {key: value.to(sample_device) for key, value in sample.items()}
        output = model.generate(**sample, max_new_tokens=100)
        logger.info(tokenizer.decode(output[0]))
        logger.info("==========================================\n\n")

        logger.info("Saving...")
        SAVE_DIR = (
            model_id.rstrip("/").split("/")[-1]
            + f"-{args.scheme}-AutoRound"
            + f"-iters{args.iters}-nsamples{args.nsamples}"
            + "-MultiGPUDDP"
            + str(world_size)
        )
        model.save_pretrained(SAVE_DIR, save_compressed=True)
        tokenizer.save_pretrained(SAVE_DIR)
        logger.info(f"Saved to {SAVE_DIR}")
