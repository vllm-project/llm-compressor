import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

import torch
from compressed_tensors.entrypoints.convert import (
    Converter,
    build_inverse_weight_maps,
    exec_jobs,
    exec_jobs_dynamic,
)
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.utils.safetensors_load import (
    InverseWeightMap,
    get_checkpoint_files,
    get_weight_map,
    is_weights_file,
    load_tensors_from_inverse_weight_map,
    update_safetensors_index,
)
from loguru import logger
from safetensors.torch import save_file

from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter
from llmcompressor.entrypoints.model_free.memory import TensorProfiler
from llmcompressor.entrypoints.model_free.save_utils import update_config
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
    :param converter: optional converter to apply to the checkpoint before
        running model-free PTQ, e.g. an AWQ or fp8 dequantizer
    """
    model_files = get_checkpoint_files(model_stub)

    config = validate_config(config, scheme, ignore)
    resolved_devices = _resolve_devices(device)
    validate_safetensors_index(model_files, config)
    os.makedirs(save_directory, exist_ok=True)

    # copy non-safetensors files (configs, tokenizers, etc.)
    for file_path, resolved_path in model_files.items():
        if not file_path.endswith("safetensors"):
            save_path = Path(save_directory) / file_path
            if is_weights_file(file_path):
                logger.warning(f"Skip processing for weights file {file_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying {file_path} -> {save_path}")
            shutil.copyfile(resolved_path, save_path)

    # build jobs without baking in a device, the scheduler assigns devices
    # dynamically based on free VRAM at submit time
    jobs = _build_jobs(model_files, save_directory, config, converter)

    # process on meta device for validation and memory estimates
    validate_jobs = [(_validate_shard, iwm, _sp, convs) for _, iwm, _sp, convs in jobs]
    memory_estimates = exec_jobs(validate_jobs, max_workers, desc="Validating")
    logger.info(
        f"Distributing {len(jobs)} shard(s), estimated memory: "
        f"{min(memory_estimates) / 1e9:.2f}-"
        f"{max(memory_estimates) / 1e9:.2f} GB per shard, "
        f"{sum(memory_estimates) / 1e9:.2f} GB total"
    )

    # quantize with dynamic GPU scheduling
    total_size = 0
    weight_map = dict()
    callable_jobs = [
        (lambda dev, fn=fn, iwm=iwm, sp=sp, convs=convs: fn(iwm, sp, convs, dev))
        for fn, iwm, sp, convs in jobs
    ]
    quantize_results = exec_jobs_dynamic(
        jobs=callable_jobs,
        devices=resolved_devices,
        max_workers=max_workers,
        memory_estimates=memory_estimates,
        desc="Quantizing",
    )
    for _total_size, _weight_map in quantize_results:
        total_size += _total_size
        weight_map.update(_weight_map)

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


def _build_jobs(
    model_files: dict[str, str],
    save_directory: str | os.PathLike,
    config: QuantizationConfig,
    converter: Converter | None,
) -> list[tuple]:
    """Build per-shard quantization jobs without baking in a device.

    Uses CT's build_inverse_weight_maps with the full converter chain so that
    ModelFreePtqConverter.get_dependencies() drives microscale partner resolution
    with no separate build_microscale_inverse_weight_maps needed.

    :returns: (jobs, memory_estimates) where each job is
        (_process_shard, inverse_weight_map, save_path, converters)
        and each memory estimate is in bytes.
    """
    weight_map = get_weight_map(model_files)

    mfptq = ModelFreePtqConverter(config)
    all_converters = ([converter] if converter is not None else []) + [mfptq]
    inverse_weight_maps = build_inverse_weight_maps(
        weight_map=weight_map,
        model_files=model_files,
        converters=all_converters,
    )

    shard_names = [name for name in model_files if name.endswith("safetensors")]

    jobs = []
    for shard_name in shard_names:
        save_path = Path(save_directory) / shard_name

        if shard_name not in inverse_weight_maps:
            raise ValueError(
                f"Could not find inverse_weight_map for shard {shard_name}"
            )

        iwm = inverse_weight_maps[shard_name]
        jobs.append((_process_shard, iwm, save_path, all_converters))

    return jobs


def _process_shard(
    inverse_weight_map: InverseWeightMap,
    save_path: str | os.PathLike,
    converters: list[Converter],
    device: torch.device,
) -> tuple[int, dict[str, str]]:
    tensors = load_tensors_from_inverse_weight_map(inverse_weight_map, device)
    for conv in converters:
        tensors = conv.process(tensors)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    save_file(tensors, save_path)
    total_size = sum(t.nbytes for t in tensors.values())
    weight_map = {key: os.path.basename(save_path) for key in tensors.keys()}
    return total_size, weight_map


def _validate_shard(
    inverse_weight_map: InverseWeightMap,
    save_path: str | os.PathLike,
    converters: list[Converter],
) -> int:
    with TensorProfiler() as prof:
        tensors = load_tensors_from_inverse_weight_map(
            inverse_weight_map, torch.device("meta")
        )
        for conv in converters:
            tensors = conv.validate(tensors)

    if prof.exception is not None or torch.device("meta") not in prof.memory_peak:
        fallback_estimate = sum(
            tensor.nbytes
            for tensor in tensors.values()
            if isinstance(tensor, torch.Tensor)
        )
        fallback_estimate = int(fallback_estimate * 2.5)
        logger.warning(
            f"Failed to estimate memory usage for {save_path}. Falling back to "
            f"2.5x size of tensor inputs ({fallback_estimate / 1e9:.2f} GB)."
        )
        return fallback_estimate

    return prof.memory_peak[torch.device("meta")]
