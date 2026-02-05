"""
python ddp_qwen3_example.py --ddp --nsamples 128 --iters 100

"""
from loguru import logger
from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
model_id = "Qwen/Qwen3-235B-A22B"
model_id = "Qwen/Qwen3-8B"
# model_id = "/data5/yiliu4/Qwen/Qwen2-0.5B"
# model_id = "Qwen/Qwen2-0.5B"

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def fix_everything(seed = 42):
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
    


def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12356")

    # Initialize process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def quantize_model(rank, world_size, args):
    """
    Quantize model on a specific GPU rank.

    Args:
        rank: GPU rank for this process
        world_size: Total number of GPUs
        args: Command line arguments
    """
    if args.deterministic:
        config_deterministic()
    logger.info(f"[Rank {rank}/{world_size}] Starting quantization")

    # Setup DDP if using multiple GPUs
    if world_size > 1:
        setup_ddp(rank, world_size)

    # Set device for this process
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Select calibration dataset.
    NUM_CALIBRATION_SAMPLES = args.nsamples
    MAX_SEQUENCE_LENGTH = 2048
    ITERS = args.iters
    # Get aligned calibration dataset.

    ds = get_dataset(
        tokenizer=tokenizer,
        seqlen=MAX_SEQUENCE_LENGTH,
        nsamples=NUM_CALIBRATION_SAMPLES,
    )

    # Configure the quantization algorithm to run.
    #   * quantize the weights to 4 bit with AutoRound with a group size 128
    #   * For `Qwen/Qwen3-235B-A22B`, it requires about 300 GB memory
    #     to run tuning with default settings.
    recipe = AutoRoundModifier(
        targets="Linear",
        scheme=args.scheme,
        ignore=[
            "lm_head",
            "re:.*mlp.gate$",
        ],
        iters=ITERS,
        enable_torch_compile=not args.disable_torch_compile,
        # device_ids="0,1,2,3",  # Use 4 A100 GPUs
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

    # Synchronize all processes
    if world_size > 1:
        dist.barrier()

    logger.info(f"[Rank {rank}] Quantization completed")
    if rank == 0:
        # Confirm generations of the quantized model look sane.
        logger.info("\n\n")
        logger.info("========== SAMPLE GENERATION ==============")
        dispatch_for_generation(model)
        sample = tokenizer("Hello my name is", return_tensors="pt")
        sample = {key: value.to(model.device) for key, value in sample.items()}
        output = model.generate(**sample, max_new_tokens=100)
        logger.info(tokenizer.decode(output[0]))
        logger.info("==========================================\n\n")

        # Save to disk compressed.
        SAVE_DIR = (
            model_name.rstrip("/").split("/")[-1]
            + f"-{args.scheme}-AutoRound"
            + f"-iters{args.iters}-nsamples{args.nsamples}"
        )
        logger.info(f"save to {SAVE_DIR}")
        model.save_pretrained(SAVE_DIR, save_compressed=True)
        tokenizer.save_pretrained(SAVE_DIR)
    else:
        # Other ranks just run quantization without saving
        logger.info(f"[Rank {rank}] Running quantization (not saving)")

    # except Exception as e:
    #     logger.info(f"[Rank {rank}] Error during quantization: {e}")
    #     raise

    # finally:
    #     # Cleanup DDP
    if world_size > 1:
        cleanup_ddp()


def main_spawn(args):
    """Main function using mp.spawn for multi-GPU quantization."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    logger.info(f"Starting DDP quantization with {world_size} GPUs")

    mp.spawn(
        quantize_model,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )

    logger.info("Quantization completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoRound Quantization with DDP support"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=model_id,
        help="Model name or path",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="W4A16",
        help="Quantization scheme (W4A16, MXFP8, MXFP4, etc.)",
    )
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--nsamples", type=int, default=256, help="Number of samples")
    parser.add_argument("--ddp", action="store_true", help="Enable DDP multi-GPU mode")
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

    # For backward compatibility with existing hardcoded values
    model_name = args.model_name

    # Parse scheme from string if needed
    from auto_round import schemes as ar_schemes

    scheme_map = {
        "FP8_STATIC": ar_schemes.FP8_STATIC,
        "MXFP8": ar_schemes.MXFP8,
        "MXFP4": ar_schemes.MXFP4,
    }
    # scheme = scheme_map.get(args.scheme, args.scheme)

    # # Check if running with torchrun
    # if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    #     logger.info("Detected torchrun environment")
    #     main_torchrun(model_name, scheme, args.iters, args.nsamples)
    if args.ddp:
        logger.info("Using mp.spawn mode for multi-GPU quantization")
        main_spawn(args)
