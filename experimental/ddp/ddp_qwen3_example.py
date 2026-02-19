#############################################################################
# This script is adapted to use DDP functionality with AutoRound.
# run this with `torchrun --nproc_per_node=2 ddp_qwen3_example.py`
# or change nproc_per_node to your desired configuration
#
# Example usage:
# torchrun --nproc_per_node=2 ddp_qwen3_example.py \
#     --model Qwen/Qwen3-8B \
#     --nsamples 128 \
#     --iters 200 \
#     --disable_torch_compile \
#     --deterministic
#############################################################################

import argparse
import os
import time

import torch
import torch.distributed as dist
from compressed_tensors.offload import dispatch_model, init_dist, load_offloaded_model
from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.autoround import AutoRoundModifier


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoRound Quantization with DDP support"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name or path",
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

###### DDP MODEL LOAD CHANGE #####
init_dist()
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map="auto_offload"
    )
##################################

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

rank = dist.get_rank()
logger.info(f"[Rank {rank}] Quantization completed")
# Confirm generations of the quantized model look sane.
logger.info("\n\n")
logger.info("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
logger.info(tokenizer.decode(output[0]))
logger.info("==========================================\n\n")

logger.info("Saving...")
# Save to disk compressed.
SAVE_DIR = (
    model_id.rstrip("/").split("/")[-1]
    + f"-{args.scheme}-AutoRound"
    + f"-iters{args.iters}-nsamples{args.nsamples}"
    + "-DDP"
    + str(dist.get_world_size())
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
logger.info(f"Saved to {SAVE_DIR}")

dist.destroy_process_group()
