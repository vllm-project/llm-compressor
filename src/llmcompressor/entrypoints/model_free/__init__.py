import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

import torch
from compressed_tensors.entrypoints.convert import (
    Converter,
    build_inverse_weight_maps,
    exec_jobs,
)
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
    is_weights_file,
    update_safetensors_index,
)
from loguru import logger

from llmcompressor.entrypoints.model_free.microscale import (
    build_microscale_inverse_weight_maps,
    has_microscale_scheme,
)
from llmcompressor.entrypoints.model_free.process import (
    process_file,
    process_file_microscale_scheme,
    validate_file,
)
from llmcompressor.entrypoints.model_free.save_utils import (
    update_config,
)
from llmcompressor.entrypoints.model_free.scheduler import (
    estimate_job_memory,
    exec_jobs_dynamic,
)
from llmcompressor.entrypoints.model_free.validate import (
    validate_config,
    validate_safetensors_index,
)

__all__ = ["model_free_ptq"]


def model_free_ptq(
    model_stub: str | os.PathLike,
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme | str | None = None,
    config: QuantizationConfig | None = None,
    ignore: Iterable[str] = tuple(),
    max_workers: int = 1,
    device: Optional[str | torch.device | list[str | torch.device]] = None,
    converter: Converter | None = None,
):
    """
    Quantize a model without the need for a model definition. This function
    operates on a model stub or folder containing weights saved in safetensors
    files.

    For microscale schemes (NVFP4, MXFP4), fused weight sets (q/k/v, gate/up)
    are handled correctly even when split across shards. Each shard job receives
    a precomputed inverse_weight_map specifying exactly which tensors to load
    from which files — enabling true partial reads with no runtime discovery
    and no redundant tensor reads.

    :param model_stub: huggingface model hub or path to local weights files
    :param save_directory: directory to save quantized weights to
    :param scheme: weight quantization scheme or preset scheme name.
        Mutually exclusive with config.
    :param config: quantization config containing one or more schemes and
        optional kv cache quantization. Mutually exclusive with scheme.
    :param ignore: modules to ignore. Modules ending with "norm" are
        automatically ignored
    :param max_workers: maximum number of concurrent worker threads.
        Effective concurrency may be lower when GPU memory is tight.
    :param device: device(s) for quantization. Accepts a single device
        string/object or a list. When multiple devices are given, shards
        are dynamically assigned based on real-time GPU memory.
    :param converter: optional converter to apply to the checkpoint to convert
        it to compressed-tensors format before running model-free PTQ
    """
    model_files = get_checkpoint_files(model_stub)

    config = validate_config(config, scheme, ignore)
    resolved_devices = _resolve_devices(device)
    validate_safetensors_index(model_files, config)
    os.makedirs(save_directory, exist_ok=True)

    moe_intermediate_size = _get_moe_intermediate_size(model_files)

    # copy non-safetensors files (configs, tokenizers, etc.)
    for file_path, resolved_path in model_files.items():
        if not file_path.endswith("safetensors"):
            save_path = Path(save_directory) / file_path
            if is_weights_file(file_path):
                logger.warning(f"Skip processing for weights file {file_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying {file_path} -> {save_path}")
            shutil.copyfile(resolved_path, save_path)

    # build jobs without baking in a device — the scheduler assigns devices
    # dynamically based on free VRAM at submit time
    jobs, mem_estimates = _build_jobs(
        model_files, save_directory, config, converter, moe_intermediate_size
    )

    # validate first (runs on meta device, so device arg is ignored)
    validate_jobs = [
        (validate_file, iwm, sp, cfg, torch.device("meta"), conv)
        for _, iwm, sp, cfg, conv, _moe_isz in jobs
    ]
    exec_jobs(validate_jobs, max_workers, desc="Validating")

    # quantize with dynamic GPU scheduling
    total_size = 0
    weight_map = dict()
    quantize_results = exec_jobs_dynamic(
        jobs=jobs,
        devices=resolved_devices,
        max_workers=max_workers,
        memory_estimates=mem_estimates,
        desc="Quantizing",
    )
    for _total_size, _weight_map in quantize_results:
        total_size += _total_size
        weight_map.update(_weight_map)

    # weight_map may contain tensors re-located to new shards (partner tensors
    # re-saved alongside the shard that needed them for fused scale computation)
    update_config(save_directory, config, converter)
    update_safetensors_index(save_directory, total_size, weight_map)


def _resolve_devices(
    device: Optional[str | torch.device | list[str | torch.device]],
) -> list[torch.device]:
    if device is None:
        count = torch.accelerator.device_count()
        if count > 0:
            devices = [torch.device(f"cuda:{i}") for i in range(count)]
            logger.info(
                f"Auto-detected {count} CUDA device(s): "
                f"{', '.join(str(d) for d in devices)}"
            )
            return devices

        logger.warning("No accelerator available! Compressing model on CPU instead")
        return [torch.device("cpu")]

    if isinstance(device, list):
        if not device:
            raise ValueError("The device list cannot be empty.")
        return [torch.device(d) for d in device]

    return [torch.device(device)]


def _get_moe_intermediate_size(
    model_files: dict[str, str],
) -> int | None:
    """Read intermediate_size from model config.json for MoE transposition detection."""
    config_path = model_files.get("config.json")
    if config_path is None:
        return None
    try:
        with open(config_path) as f:
            model_config = json.load(f)
        return model_config.get("intermediate_size")
    except (OSError, json.JSONDecodeError):
        return None


def _build_jobs(
    model_files: dict[str, str],
    save_directory: str | os.PathLike,
    config: QuantizationConfig,
    converter: Converter | None,
    moe_intermediate_size: int | None = None,
) -> tuple[list[tuple], list[int]]:
    """Build per-shard quantization jobs **without** device assignment.

    Device selection is deferred to ``exec_jobs_dynamic`` which picks a GPU
    at submit time based on real-time free VRAM.

    :returns: ``(jobs, memory_estimates)`` — each job is a tuple of
        ``(fn, inverse_weight_map, save_path, config, converter,
        moe_intermediate_size)`` and each memory estimate is in bytes.
    """
    weight_map = get_weight_map(model_files)

    if has_microscale_scheme(config):
        job_fn = process_file_microscale_scheme
        build_iwm_fn = build_microscale_inverse_weight_maps
    else:
        job_fn = process_file
        build_iwm_fn = build_inverse_weight_maps

    inverse_weight_maps = build_iwm_fn(
        weight_map=weight_map,
        model_files=model_files,
        converters=[converter] if converter is not None else [],
    )

    shard_names = [name for name in model_files if name.endswith("safetensors")]

    jobs = []
    mem_estimates = []
    for shard_name in shard_names:
        save_path = Path(save_directory) / shard_name

        if shard_name not in inverse_weight_maps:
            raise ValueError(
                f"Could not find inverse_weight_map for shard {shard_name}"
            )

        iwm = inverse_weight_maps[shard_name]
        jobs.append((job_fn, iwm, save_path, config, converter, moe_intermediate_size))
        mem_estimates.append(estimate_job_memory(iwm))

    if mem_estimates:
        logger.info(
            f"Distributing {len(jobs)} shard(s), estimated memory: "
            f"{min(mem_estimates) / 1e9:.2f}-"
            f"{max(mem_estimates) / 1e9:.2f} GB per shard, "
            f"{sum(mem_estimates) / 1e9:.2f} GB total"
        )
        logger.debug(
            "Per-shard estimates: "
            + ", ".join(f"{m / 1e9:.2f} GB" for m in mem_estimates)
        )

    return jobs, mem_estimates
