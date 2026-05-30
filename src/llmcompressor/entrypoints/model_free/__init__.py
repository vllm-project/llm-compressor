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
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
    is_weights_file,
    update_safetensors_index,
)
from loguru import logger

from llmcompressor.entrypoints.model_free.microscale import (
    build_microscale_inverse_weight_maps,
    is_microscale_scheme,
)
from llmcompressor.entrypoints.model_free.process import (
    process_file,
    process_file_microscale_scheme,
    validate_file,
)
from llmcompressor.entrypoints.model_free.save_utils import (
    update_config,
)
from llmcompressor.entrypoints.model_free.validate import (
    validate_safetensors_index,
    validate_scheme,
)

__all__ = ["model_free_ptq"]


def model_free_ptq(
    model_stub: str | os.PathLike,
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme | str,
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
    :param scheme: weight quantization scheme or preset scheme name
    :param ignore: modules to ignore. Modules ending with "norm" are
        automatically ignored
    :param max_workers: number of worker threads to process files with
    :param device: gpu devices to accelerate quantization with.
    :param converter: optional converter to apply to the checkpoint to convert
        it to compressed-tensors format before running model-free PTQ
    """
    # validate arguments
    model_files = get_checkpoint_files(model_stub)

    scheme_name, scheme = validate_scheme(scheme)
    resolved_devices = _resolve_devices(device)
    validate_safetensors_index(model_files, scheme)

    # copy non-safetensors files (configs, tokenizers, etc.)
    for file_path, resolved_path in model_files.items():
        if not file_path.endswith("safetensors"):
            save_path = Path(save_directory) / file_path
            if is_weights_file(file_path):
                logger.warning(f"Skip processing for weights file {file_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying {file_path} -> {save_path}")
            shutil.copyfile(resolved_path, save_path)

    # build quantization jobs
    jobs = _build_jobs(
        model_files, save_directory, scheme, ignore, resolved_devices, converter
    )

    # 1. validate quantizable tensors — fail fast before long-running quantization
    validate_jobs = [(validate_file, *job[1:]) for job in jobs]
    exec_jobs(validate_jobs, max_workers, desc="Validating")

    # 2-5. quantize and compress weights
    total_size = 0
    weight_map = dict()
    quantize_results = exec_jobs(jobs, max_workers, desc="Quantizing")
    for _total_size, _weight_map in quantize_results:
        total_size += _total_size
        weight_map.update(_weight_map)

    # 6. update config and safetensors index
    # weight_map may contain tensors re-located to new shards (partner tensors
    # re-saved alongside the shard that needed them for fused scale computation)
    update_config(save_directory, scheme_name, scheme, ignore, converter)
    update_safetensors_index(save_directory, total_size, weight_map)


def _resolve_devices(
    device: Optional[str | torch.device | list[str | torch.device]],
) -> list[torch.device]:
    if device is None:
        count = torch.cuda.device_count()
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


def _build_jobs(
    model_files: dict[str, str],
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    devices: list[torch.device],
    converter: Converter | None,
) -> list[tuple]:
    """
    Build jobs with precomputed inverse_weight_map per shard.

    For each output shard, build_inverse_weight_map() determines exactly which
    tensors to load from which source files — including any fused partner tensors
    from other shards. This avoids runtime fused-partner discovery inside the
    process function and eliminates redundant tensor reads.

    :returns: list of jobs tuples
        (job_fn, inverse_weight_map, save_path, scheme, ignore, device, converter)
        Shards are distributed round-robin across the given devices.
    """
    weight_map = get_weight_map(model_files)

    if is_microscale_scheme(scheme):
        job_fn = process_file_microscale_scheme
        build_inverse_weight_maps_fn = build_microscale_inverse_weight_maps
    else:
        job_fn = process_file
        build_inverse_weight_maps_fn = build_inverse_weight_maps

    inverse_weight_maps = build_inverse_weight_maps_fn(
        weight_map=weight_map,
        model_files=model_files,
        converters=[converter] if converter is not None else [],
    )

    shard_names = [name for name in model_files if name.endswith("safetensors")]
    logger.info(
        f"Distributing {len(shard_names)} shard(s) across {len(devices)} "
        f"device(s): {', '.join(str(d) for d in devices)}"
    )

    jobs = []
    for i, shard_name in enumerate(shard_names):
        save_path = Path(save_directory) / shard_name

        if shard_name not in inverse_weight_maps:
            raise ValueError(
                f"Could not find inverse_weight_map for shard {shard_name}"
            )

        device = devices[i % len(devices)]

        jobs.append(
            (
                job_fn,
                inverse_weight_maps[shard_name],
                save_path,
                scheme,
                ignore,
                device,
                converter,
            )
        )

    return jobs
