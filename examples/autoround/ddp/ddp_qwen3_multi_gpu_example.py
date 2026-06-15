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
import importlib
import os
import sys
import time
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
from compressed_tensors.offload import dispatch_model, from_accelerate, load_offloaded_model
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot


class StopAfterBlocks(RuntimeError):
    pass


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


def _module_has_direct_tensors(module: torch.nn.Module) -> bool:
    return any(t is not None for t in module._parameters.values()) or any(
        t is not None for t in module._buffers.values()
    )


def _module_has_meta_tensors(module: torch.nn.Module) -> bool:
    return any(
        t is not None and t.device.type == "meta"
        for t in module._parameters.values()
    ) or any(t is not None and t.device.type == "meta" for t in module._buffers.values())


def patch_ct_dispatch_for_sparse_offload():
    """Avoid wrapping modules that do not need compressed-tensors offload hooks."""
    dispatch_mod = importlib.import_module("compressed_tensors.offload.dispatch")
    fa_mod = importlib.import_module("compressed_tensors.offload.convert.from_accelerate")

    if getattr(dispatch_mod.dispatch_with_map, "_llmc_sparse_patch", False):
        return

    offload_module = dispatch_mod.offload_module
    tqdm = dispatch_mod.tqdm

    def optimized_dispatch_with_map(
        model: torch.nn.Module,
        device_map,
        offload_dir: str | None = None,
        show_progress: bool = True,
    ):
        filtered = []
        skipped_noop = 0
        skipped_empty = 0
        skipped_cpu_cpu = 0
        kept_meta_materialization = 0

        for name, (onload_device, offload_device) in device_map.items():
            if offload_device is None:
                skipped_noop += 1
                continue

            module = model.get_submodule(name)
            if not _module_has_direct_tensors(module):
                skipped_empty += 1
                continue

            if (
                str(onload_device) == "cpu"
                and str(offload_device) == "cpu"
                and not _module_has_meta_tensors(module)
            ):
                skipped_cpu_cpu += 1
                continue

            if str(onload_device) == "cpu" and str(offload_device) == "cpu":
                kept_meta_materialization += 1

            filtered.append((name, onload_device, offload_device))

        logger.info(
            "Compressed-tensors dispatch filtered {} -> {} modules "
            "(noop={}, empty={}, cpu_to_cpu_skipped={}, cpu_to_cpu_meta_kept={})",
            len(device_map),
            len(filtered),
            skipped_noop,
            skipped_empty,
            skipped_cpu_cpu,
            kept_meta_materialization,
        )

        for name, onload_device, offload_device in tqdm(
            filtered,
            desc="Dispatching model",
            disable=(not show_progress),
        ):
            module = model.get_submodule(name)
            if offload_device == "disk":
                offload_module(
                    module,
                    onload_device,
                    offload_device,
                    offload_dir=offload_dir,
                )
            else:
                offload_module(module, onload_device, offload_device)

    optimized_dispatch_with_map._llmc_sparse_patch = True
    dispatch_mod.dispatch_with_map = optimized_dispatch_with_map
    fa_mod.dispatch_with_map = optimized_dispatch_with_map


def _rank_offload_folder(base_folder: str | None) -> str | None:
    if not base_folder:
        return None

    rank, _ = get_dist_info()
    rank_folder = Path(base_folder) / f"rank{rank}"
    rank_folder.mkdir(parents=True, exist_ok=True)
    return str(rank_folder)


def _independent_cpu_max_memory(extra_cpu_mem: int = int(5e9)) -> dict[str, int]:
    _, world_size = get_dist_info()
    per_rank_available = psutil.virtual_memory().available // max(world_size, 1)
    return {"cpu": max(per_rank_available - extra_cpu_mem, int(8e9))}


def load_model_with_local_offload(model_id: str, offload_folder: str | None):
    """Load model on each rank independently, then convert accelerate offload locally."""
    load_kwargs = {
        "dtype": "auto",
        "device_map": "auto",
        "max_memory": _independent_cpu_max_memory(),
    }
    rank_offload_folder = _rank_offload_folder(offload_folder)
    if rank_offload_folder:
        load_kwargs["offload_folder"] = rank_offload_folder

    logger.info(
        "[Rank {}] Loading model independently with max_memory={} offload_folder={}",
        get_dist_info()[0],
        load_kwargs["max_memory"],
        rank_offload_folder,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if hasattr(model, "hf_device_map"):
        from_accelerate(model)
    return model


def patch_disable_onloading_for_quant_init():
    """Avoid expensive dist.broadcast + barrier for every new quant param.

    When DDP is initialized before model loading,
    ``OffloadCache.cls_from_device`` selects distributed cache variants
    (DistributedCPUCache / DistributedDiskCache).  Each call to
    ``register_parameter`` inside ``initialize_module_for_quantization``
    triggers ``offload()``, which does a collective broadcast + barrier.
    For large MoE models (e.g. Qwen3-235B with 100K+ Linear layers × 6
    quant params), this means hundreds of thousands of round-trips —
    effectively hanging the process.

    Wrapping the body in ``disable_onloading()`` stores new parameters
    directly in ``offloaded_values`` without invoking the distributed
    offload, cutting the overhead to zero.
    """
    from compressed_tensors.offload import (  # noqa: F811
        disable_onloading,
    )

    lifecycle_init_mod = importlib.import_module(
        "compressed_tensors.quantization.lifecycle.initialize"
    )
    original_fn = lifecycle_init_mod.initialize_module_for_quantization

    if getattr(original_fn, "_llmc_no_dist_offload_patch", False):
        return

    def patched_initialize_module_for_quantization(module, scheme=None, force_zero_point=True):
        with disable_onloading():
            return original_fn(module, scheme=scheme, force_zero_point=force_zero_point)

    patched_initialize_module_for_quantization._llmc_no_dist_offload_patch = True
    lifecycle_init_mod.initialize_module_for_quantization = (
        patched_initialize_module_for_quantization
    )


def patch_autoround_stop_after_blocks(max_blocks: int):
    """Raise after N decoding blocks finish so large-model smoke tests can stop cleanly."""
    autoround_mod = importlib.import_module("llmcompressor.modifiers.autoround.base")
    modifier_cls = autoround_mod.AutoRoundModifier

    if getattr(modifier_cls.apply_autoround, "_llmc_stop_after_patch", False):
        return

    original_apply_autoround = modifier_cls.apply_autoround

    def wrapped_apply_autoround(self, state, modules):
        modules = modules or []
        if not any(self._is_decoding_layer(module) for module in modules):
            return original_apply_autoround(self, state, modules)

        result = original_apply_autoround(self, state, modules)
        completed = getattr(self, "_llmc_completed_blocks", 0) + 1
        self._llmc_completed_blocks = completed
        logger.info(
            "[Rank {}] Completed AutoRound block {}/{}",
            get_dist_info()[0],
            completed,
            max_blocks,
        )
        if completed >= max_blocks:
            raise StopAfterBlocks(f"Stopped after {completed} blocks")
        return result

    wrapped_apply_autoround._llmc_stop_after_patch = True
    modifier_cls.apply_autoround = wrapped_apply_autoround


def patch_llmc_timing_logs():
    """Add coarse timing logs around the expensive LLMC startup stages."""
    recipe_mod = importlib.import_module("llmcompressor.recipe.recipe")
    lifecycle_mod = importlib.import_module("llmcompressor.core.lifecycle")
    quant_mixin_mod = importlib.import_module(
        "llmcompressor.modifiers.quantization.quantization.mixin"
    )
    quantization_base_mod = importlib.import_module(
        "compressed_tensors.quantization"
    )
    module_utils_mod = importlib.import_module("compressed_tensors.utils")
    group_validation_mod = importlib.import_module(
        "llmcompressor.modifiers.quantization.group_size_validation"
    )
    seq_helpers_mod = importlib.import_module("llmcompressor.pipelines.sequential.helpers")
    seq_pipeline_mod = importlib.import_module("llmcompressor.pipelines.sequential.pipeline")
    cache_mod = importlib.import_module("llmcompressor.pipelines.cache")
    autoround_mod = importlib.import_module("llmcompressor.modifiers.autoround.base")
    core_mod = importlib.import_module("llmcompressor.core")

    recipe_cls = recipe_mod.Recipe
    lifecycle_cls = lifecycle_mod.CompressionLifecycle
    quant_mixin_cls = quant_mixin_mod.QuantizationMixin
    cache_cls = cache_mod.IntermediatesCache
    autoround_cls = autoround_mod.AutoRoundModifier
    seq_pipeline_cls = seq_pipeline_mod.SequentialPipeline
    lifecycle_callbacks = core_mod.LifecycleCallbacks

    if getattr(recipe_cls.from_modifiers, "_llmc_timing_patch", False):
        return

    original_from_modifiers = recipe_cls.from_modifiers.__func__
    original_lifecycle_initialize = lifecycle_cls.initialize
    original_initialize_quantization = quant_mixin_cls.initialize_quantization
    original_start_calibration = autoround_cls.start_calibration
    original_trace_subgraphs = seq_helpers_mod.trace_subgraphs
    original_from_dataloader = cache_cls.from_dataloader.__func__
    original_apply_autoround = autoround_cls.apply_autoround
    original_seq_call = seq_pipeline_cls.__call__
    original_calib_epoch_start = lifecycle_callbacks.calibration_epoch_start
    original_match_named_modules = module_utils_mod.match_named_modules
    original_apply_quantization_config = quantization_base_mod.apply_quantization_config
    original_validate_group_size_divisibility = (
        group_validation_mod.validate_group_size_divisibility
    )

    def _timed(label, fn, *args, **kwargs):
        start = time.perf_counter()
        logger.info("[Rank {}] {} started", get_dist_info()[0], label)
        try:
            return fn(*args, **kwargs)
        finally:
            logger.info(
                "[Rank {}] {} finished in {:.2f}s",
                get_dist_info()[0],
                label,
                time.perf_counter() - start,
            )

    @classmethod
    def timed_from_modifiers(cls, modifiers, modifier_group_name=None):
        return _timed(
            "Recipe.from_modifiers",
            original_from_modifiers,
            cls,
            modifiers,
            modifier_group_name,
        )

    def timed_lifecycle_initialize(self, *args, **kwargs):
        return _timed(
            "CompressionLifecycle.initialize",
            original_lifecycle_initialize,
            self,
            *args,
            **kwargs,
        )

    def timed_initialize_quantization(self, model):
        return _timed(
            "QuantizationMixin.initialize_quantization",
            original_initialize_quantization,
            self,
            model,
        )

    def timed_start_calibration(self, model):
        return _timed(
            "AutoRoundModifier.start_calibration",
            original_start_calibration,
            self,
            model,
        )

    def timed_trace_subgraphs(*args, **kwargs):
        return _timed("trace_subgraphs", original_trace_subgraphs, *args, **kwargs)

    @classmethod
    def timed_from_dataloader(cls, *args, **kwargs):
        return _timed(
            "IntermediatesCache.from_dataloader",
            original_from_dataloader,
            cls,
            *args,
            **kwargs,
        )

    def timed_apply_autoround(self, state, modules):
        modules = modules or []
        decoding_layers = [m for m in modules if self._is_decoding_layer(m)]
        if not decoding_layers:
            return original_apply_autoround(self, state, modules)
        layer_name = getattr(decoding_layers[0], "_tmp_name", decoding_layers[0].__class__.__name__)
        return _timed(
            f"AutoRoundModifier.apply_autoround({layer_name})",
            original_apply_autoround,
            self,
            state,
            modules,
        )

    def timed_seq_call(model, dataloader, dataset_args):
        pipeline_start = time.perf_counter()
        logger.info("[Rank {}] SequentialPipeline.__call__ started", get_dist_info()[0])
        try:
            logger.info("[Rank {}] SequentialPipeline pre-next(iter(dataloader))", get_dist_info()[0])
            iter_start = time.perf_counter()
            sample_input = next(iter(dataloader))
            logger.info(
                "[Rank {}] next(iter(dataloader)) finished in {:.2f}s",
                get_dist_info()[0],
                time.perf_counter() - iter_start,
            )
            del sample_input
            return original_seq_call(model, dataloader, dataset_args)
        finally:
            logger.info(
                "[Rank {}] SequentialPipeline.__call__ finished in {:.2f}s",
                get_dist_info()[0],
                time.perf_counter() - pipeline_start,
            )

    def timed_calib_epoch_start(*args, **kwargs):
        return _timed(
            "LifecycleCallbacks.calibration_epoch_start",
            original_calib_epoch_start,
            *args,
            **kwargs,
        )

    def timed_match_named_modules(*args, **kwargs):
        return _timed("match_named_modules", original_match_named_modules, *args, **kwargs)

    def timed_apply_quantization_config(*args, **kwargs):
        return _timed(
            "apply_quantization_config",
            original_apply_quantization_config,
            *args,
            **kwargs,
        )

    def timed_validate_group_size_divisibility(*args, **kwargs):
        return _timed(
            "validate_group_size_divisibility",
            original_validate_group_size_divisibility,
            *args,
            **kwargs,
        )

    timed_from_modifiers._llmc_timing_patch = True
    recipe_cls.from_modifiers = timed_from_modifiers
    lifecycle_cls.initialize = timed_lifecycle_initialize
    quant_mixin_cls.initialize_quantization = timed_initialize_quantization
    autoround_cls.start_calibration = timed_start_calibration
    module_utils_mod.match_named_modules = timed_match_named_modules
    quant_mixin_mod.match_named_modules = timed_match_named_modules
    quantization_base_mod.apply_quantization_config = timed_apply_quantization_config
    quant_mixin_mod.apply_quantization_config = timed_apply_quantization_config
    group_validation_mod.validate_group_size_divisibility = timed_validate_group_size_divisibility
    quant_mixin_mod.validate_group_size_divisibility = timed_validate_group_size_divisibility
    seq_helpers_mod.trace_subgraphs = timed_trace_subgraphs
    seq_pipeline_mod.trace_subgraphs = timed_trace_subgraphs
    cache_cls.from_dataloader = timed_from_dataloader
    autoround_cls.apply_autoround = timed_apply_autoround
    seq_pipeline_cls.__call__ = staticmethod(timed_seq_call)
    lifecycle_callbacks.calibration_epoch_start = timed_calib_epoch_start


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
    parser.add_argument("--iters", type=int, default=20, help="Number of iterations")
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
    parser.add_argument(
        "--offload-folder",
        type=str,
        default=None,
        help="Optional folder for disk offload while loading very large models",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Optional number of decoder blocks to quantize before exiting",
    )
    args = parser.parse_args()

    if args.deterministic:
        config_deterministic()

    model_id = args.model

    ###### MULTI-GPU DDP INIT #####
    init_dist_multi_gpu(gpus_per_group=args.gpus_per_group)
    patch_ct_dispatch_for_sparse_offload()
    patch_llmc_timing_logs()
    patch_disable_onloading_for_quant_init()
    if args.max_blocks is not None:
        patch_autoround_stop_after_blocks(args.max_blocks)
    # Load onto CPU first and spill to disk if needed. AutoRound will then
    # onload and shard each block onto the rank-local GPU group during tuning.
    load_start = time.perf_counter()
    rank, world_size = get_dist_info()
    if world_size > 1:
        model = load_model_with_local_offload(model_id, args.offload_folder)
    else:
        load_kwargs = {
            "dtype": "auto",
            "device_map": "auto_offload",
        }
        rank_offload_folder = _rank_offload_folder(args.offload_folder)
        if rank_offload_folder:
            load_kwargs["offload_folder"] = rank_offload_folder
        with load_offloaded_model():
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    logger.info(
        "[Rank {}] Model load + offload conversion finished in {:.2f}s",
        rank,
        time.perf_counter() - load_start,
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
    stopped_early = False
    try:
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            shuffle_calibration_samples=False,
        )
    except StopAfterBlocks as exc:
        stopped_early = True
        logger.info("[Rank {}] {}", get_dist_info()[0], str(exc))

    rank, world_size = get_dist_info()
    if stopped_early:
        logger.info(f"[Rank {rank}] Partial quantization completed")
    else:
        logger.info(f"[Rank {rank}] Quantization completed")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if rank != 0:
        sys.exit(0)

    if stopped_early:
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
