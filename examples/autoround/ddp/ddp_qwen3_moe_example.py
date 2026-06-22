"""
DDP AutoRound quantization example for large MoE models.

Runs 2 ranks, each using GPUS_PER_GROUP GPUs. All ranks load the model
independently on CPU (safetensors mmap shares physical pages at OS level).

Run with:
  CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_GROUP=2 torchrun \
    --nproc_per_node=2 ddp_qwen3_moe_example.py
"""

import os
import time

import torch
import torch.distributed as dist
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from compressed_tensors.offload.cache.base import force_local_cache

MODEL = "/storage/yiliu7/Qwen/Qwen3-235B-A22B-Instruct-2507"
SCHEME = "W4A16"
ITERS = 100
NSAMPLES = 256

###### DDP INIT #####
gpus_per_group = int(os.environ.get("GPUS_PER_GROUP", "1"))
if "TORCHELASTIC_RUN_ID" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    main_gpu = local_rank * gpus_per_group
    torch.cuda.set_device(main_gpu)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{main_gpu}"),
    )

rank = dist.get_rank() if dist.is_initialized() else 0
world_size = dist.get_world_size() if dist.is_initialized() else 1
main_gpu = rank * gpus_per_group
logger.info(
    f"[Rank {rank}/{world_size}] GPUs: {torch.cuda.device_count()}, "
    f"main_gpu: {main_gpu}, group: [{main_gpu}-{main_gpu + gpus_per_group - 1}]"
)

###### MODEL LOAD #####
load_start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto")
load_elapsed = time.perf_counter() - load_start
logger.info(f"[Rank {rank}] Model loaded on CPU in {load_elapsed:.1f}s")

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
with force_local_cache():
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=NSAMPLES,
        shuffle_calibration_samples=False,
    )
quant_elapsed = time.perf_counter() - quant_start
logger.info(f"[Rank {rank}] Quantization done in {quant_elapsed:.1f}s")

if dist.is_initialized():
    dist.barrier()

###### SAVE (rank 0 only) #####
if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()

if rank == 0:
    save_dir = (
         "/storage/yiliu7/Qwen/"
         + MODEL.rstrip("/").split("/")[-1]
        + f"-{SCHEME}-AutoRound"
        + f"-iters{ITERS}-nsamples{NSAMPLES}"
        + f"-DDP{world_size}"
    )
    logger.info(f"Saving to {save_dir}...")
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Saved to {save_dir}")

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
