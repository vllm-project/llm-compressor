"""
python ddp_qwen3_example.py --ddp --nsamples 128 --iters 100

"""

from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
model_id = "Qwen/Qwen3-235B-A22B"
model_id = "Qwen/Qwen3-8B"
model_id = "/data5/yiliu4/Qwen/Qwen2-0.5B"

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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


def quantize_model(rank, world_size, model_name, scheme, iters=4, nsamples=32):
    """
    Quantize model on a specific GPU rank.

    Args:
        rank: GPU rank for this process
        world_size: Total number of GPUs
        model_name: Model name or path
        scheme: Quantization scheme
        iters: Number of iterations
        nsamples: Number of samples
    """
    print(f"[Rank {rank}/{world_size}] Starting quantization")

    # Setup DDP if using multiple GPUs
    if world_size > 1:
        setup_ddp(rank, world_size)

    # Set device for this process
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Select calibration dataset.
    NUM_CALIBRATION_SAMPLES = nsamples
    MAX_SEQUENCE_LENGTH = 2048
    ITERS = iters
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
        scheme="W4A16",
        ignore=[
            "lm_head",
            "re:.*mlp.gate$",
        ],
        iters=ITERS,
        enable_torch_compile=False,
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

    print(f"[Rank {rank}] Quantization completed")
    if rank == 0:
        # Confirm generations of the quantized model look sane.
        print("\n\n")
        print("========== SAMPLE GENERATION ==============")
        dispatch_for_generation(model)
        sample = tokenizer("Hello my name is", return_tensors="pt")
        sample = {key: value.to(model.device) for key, value in sample.items()}
        output = model.generate(**sample, max_new_tokens=100)
        print(tokenizer.decode(output[0]))
        print("==========================================\n\n")

        # Save to disk compressed.
        SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128-AutoRound"
        print(f"save to {SAVE_DIR}")
        model.save_pretrained(SAVE_DIR, save_compressed=True)
        tokenizer.save_pretrained(SAVE_DIR)
    else:
        # Other ranks just run quantization without saving
        print(f"[Rank {rank}] Running quantization (not saving)")

    # except Exception as e:
    #     print(f"[Rank {rank}] Error during quantization: {e}")
    #     raise

    # finally:
    #     # Cleanup DDP
    if world_size > 1:
        cleanup_ddp()


def main_spawn(model_name, scheme, iters, nsamples):
    """Main function using mp.spawn for multi-GPU quantization."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # if world_size < 2:
    #     print("Warning: Only 1 GPU detected. Running single GPU mode.")
    #     return main_single_gpu(model_name, scheme, iters, nsamples)
    print(f"Starting DDP quantization with {world_size} GPUs")

    mp.spawn(
        quantize_model,
        args=(world_size, model_name, scheme, iters, nsamples),
        nprocs=world_size,
        join=True,
    )

    print("Quantization completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoRound Quantization with DDP support"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/data5/yiliu4/Qwen/Qwen2-0.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="FP8_STATIC",
        help="Quantization scheme (FP8_STATIC, MXFP8, MXFP4, etc.)",
    )
    parser.add_argument("--iters", type=int, default=4, help="Number of iterations")
    parser.add_argument("--nsamples", type=int, default=32, help="Number of samples")
    parser.add_argument("--ddp", action="store_true", help="Enable DDP multi-GPU mode")

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
    #     print("Detected torchrun environment")
    #     main_torchrun(model_name, scheme, args.iters, args.nsamples)
    if args.ddp:
        print("Using mp.spawn mode for multi-GPU quantization")
        main_spawn(model_name, args.scheme, args.iters, args.nsamples)
