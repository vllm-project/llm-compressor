"""
DDP AutoRound quantization example for large MoE models.

Runs 2 ranks, each using GPUS_PER_GROUP GPUs. All ranks load the model
independently on CPU (safetensors mmap shares physical pages at OS level).

Run with:
  CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_GROUP=2 torchrun \
    --nproc_per_node=2 ddp_autoround.py \
    --model /storage/yiliu7/Qwen/Qwen3-235B-A22B-Instruct-2507/ 2>&1 | tee test_ddp_autoround.log
  CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_GROUP=2 torchrun \
    --nproc_per_node=2 ddp_autoround.py \
    --model /storage/yiliu7/Qwen/Qwen3-30B-A3B-Instruct-2507/ 2>&1 | tee test_ddp_autoround.log
  CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_GROUP=2 torchrun \
    --nproc_per_node=2 ddp_autoround.py \
    --model /path/to/model
"""

import argparse
import importlib
import os
import time

import torch
import torch.distributed as dist
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot


def patch_disable_onloading_for_quant_init():
    """Avoid dist.broadcast + barrier for every new quant parameter.

    compressed-tensors' initialize_module_for_quantization creates new
    parameters which trigger DistributedCPUCache's per-param broadcast.
    Wrapping with disable_onloading() prevents this.
    """
    from compressed_tensors.offload import disable_onloading

    lifecycle_init_mod = importlib.import_module(
        "compressed_tensors.quantization.lifecycle.initialize"
    )
    original_fn = lifecycle_init_mod.initialize_module_for_quantization
    if getattr(original_fn, "_patched", False):
        return

    def patched(module, scheme=None, force_zero_point=True):
        with disable_onloading():
            return original_fn(module, scheme=scheme, force_zero_point=force_zero_point)

    patched._patched = True
    lifecycle_init_mod.initialize_module_for_quantization = patched


def patch_force_local_cache():
    """Force OffloadCache.cls_from_device to return non-distributed caches.

    When ranks load the model independently, each already has parameters
    locally. DistributedCPUCache's per-param broadcast+barrier is
    unnecessary and causes O(n_params) collective ops (~218ms each).
    """
    from compressed_tensors.offload.cache.base import OffloadCache
    from compressed_tensors.offload.cache.cpu import CPUCache
    from compressed_tensors.offload.cache.device import DeviceCache
    from compressed_tensors.offload.cache.disk import DiskCache
    from compressed_tensors.utils import is_accelerator_type

    @classmethod
    def cls_from_device_local(cls, device=None):
        device_type = torch.device(device).type if device != "disk" else "disk"
        if device_type == "cpu":
            return CPUCache
        elif is_accelerator_type(device_type):
            return DeviceCache
        elif device_type == "disk":
            return DiskCache
        else:
            raise NotImplementedError(f"Offload of type {device_type} not implemented")

    OffloadCache.cls_from_device = cls_from_device_local
    logger.info("Patched OffloadCache.cls_from_device → local (non-distributed) caches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--scheme", type=str, default="W4A16")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--nsamples", type=int, default=128)
    args = parser.parse_args()

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

    # Apply patches BEFORE model loading and calibration
    patch_disable_onloading_for_quant_init()
    patch_force_local_cache()

    ###### MODEL LOAD #####
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    load_elapsed = time.perf_counter() - load_start
    logger.info(f"[Rank {rank}] Model loaded on CPU in {load_elapsed:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ###### DATASET #####
    os.environ["AR_DISABLE_DATASET_SUBPROCESS"] = "1"
    from auto_round.calib_dataset import get_dataset
    from llmcompressor.modifiers.autoround import AutoRoundModifier

    ds = get_dataset(tokenizer=tokenizer, seqlen=2048, nsamples=args.nsamples)

    ###### RECIPE #####
    recipe = AutoRoundModifier(
        targets="Linear",
        scheme=args.scheme,
        ignore=["lm_head", "re:.*mlp.gate$"],
        iters=args.iters,
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
        num_calibration_samples=args.nsamples,
        shuffle_calibration_samples=False,
    )
    quant_elapsed = time.perf_counter() - quant_start
    logger.info(f"[Rank {rank}] Quantization done in {quant_elapsed:.1f}s")

    if dist.is_initialized():
        dist.barrier()

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

    ###### SAVE (rank 0 only) #####
    if rank == 0:
        save_dir = (
            args.model.rstrip("/").split("/")[-1]
            + f"-{args.scheme}-AutoRound"
            + f"-iters{args.iters}-nsamples{args.nsamples}"
            + f"-DDP{world_size}"
        )
        logger.info(f"Saving to {save_dir}...")
        model.save_pretrained(save_dir, save_compressed=True)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved to {save_dir}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logger.info(f"[Rank {rank}] SUCCESS")
