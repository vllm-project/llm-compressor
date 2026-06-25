"""
DDP AutoRound quantization example for large MoE models.

Uses the standard compressed-tensors DDP path: load_offloaded_model()
broadcasts weights from rank 0 to rank 1.  GPUS_PER_GROUP controls how
many GPUs each rank uses for per-block model parallelism.

Run with:
  CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_GROUP=2 torchrun \
    --nproc_per_node=2 ddp_qwen3_moe_example.py
"""

import os
import time

import torch
import torch.distributed as dist
from compressed_tensors.offload import load_offloaded_model
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot

MODEL = "/storage/yiliu7/Qwen/Qwen3-235B-A22B-Instruct-2507"
SCHEME = "W4A16"
ITERS = 1
NSAMPLES = 4

###### DDP INIT #####
gpus_per_group = int(os.environ.get("GPUS_PER_GROUP", "1"))
local_rank = int(os.environ["LOCAL_RANK"])
main_gpu = local_rank * gpus_per_group
torch.cuda.set_device(main_gpu)
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    device_id=torch.device(f"cuda:{main_gpu}"),
)

rank = dist.get_rank()
world_size = dist.get_world_size()
logger.info(
    f"[Rank {rank}/{world_size}] GPUs: {torch.cuda.device_count()}, "
    f"main_gpu: {main_gpu}, group: [{main_gpu}-{main_gpu + gpus_per_group - 1}]"
)

###### MODEL LOAD #####
load_start = time.perf_counter()
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype="auto", device_map="auto_offload",
    )
logger.info(f"[Rank {rank}] Loaded in {time.perf_counter() - load_start:.1f}s")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

###### DATASET #####
os.environ["AR_DISABLE_DATASET_SUBPROCESS"] = "1"
from auto_round.calib_dataset import get_dataset
from llmcompressor.modifiers.autoround import AutoRoundModifier

ds = get_dataset(tokenizer=tokenizer, seqlen=2048, nsamples=NSAMPLES)

###### RECIPE #####

recipe = AutoRoundModifier(
    targets="Linear",
    scheme=SCHEME,
    ignore=["lm_head", "re:.*mlp.gate$"],
    iters=ITERS,
    enable_torch_compile=False,
)

###### QUANTIZE #####
logger.info(f"[Rank {rank}] Starting oneshot...")
quant_start = time.perf_counter()
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=NSAMPLES,
    shuffle_calibration_samples=False,
)
logger.info(f"[Rank {rank}] Quantization done in {time.perf_counter() - quant_start:.1f}s")

###### SAVE #####
# Both ranks must participate — save_pretrained internally calls
# collectives (broadcast_object_list). Only rank 0 writes to disk.
save_dir = (
    MODEL.rstrip("/").split("/")[-1]
    + f"-{SCHEME}-AutoRound"
    + f"-iters{ITERS}-nsamples{NSAMPLES}"
    + f"-DDP{world_size}"
)
logger.info(f"[Rank {rank}] Saving to {save_dir}...")
model.save_pretrained(save_dir, save_compressed=True)
if rank == 0:
    tokenizer.save_pretrained(save_dir)
logger.info(f"[Rank {rank}] Saved to {save_dir}")

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()

###### SAMPLE GENERATION (rank 0 only) #####
if rank == 0:
    from compressed_tensors.offload import dispatch_model

    logger.info("========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    sample = tokenizer("Hello my name is", return_tensors="pt")
    sample = {key: value.to(model.device) for key, value in sample.items()}
    output = model.generate(**sample, max_new_tokens=100)
    logger.info(tokenizer.decode(output[0]))
    logger.info("==========================================")

logger.info(f"[Rank {rank}] SUCCESS")
