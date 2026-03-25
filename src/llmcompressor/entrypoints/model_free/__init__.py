import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

import torch
from compressed_tensors.entrypoints.convert import (
    Converter,
    exec_jobs,
)
from compressed_tensors.entrypoints.convert.file_utils import (
    get_checkpoint_files,
    is_weights_file,
)
from compressed_tensors.quantization import QuantizationScheme
from loguru import logger



from llmcompressor.entrypoints.model_free.helpers import (
    find_safetensors_index_file,
    gpu_if_available,
)
from llmcompressor.entrypoints.model_free.microscale import (
    build_inverse_weights_map,
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
    device: Optional[torch.device | str] = None,
    converter: Converter | None = None,
):
    """
    Quantize a model without the need for a model definition. This function
    operates on a model stub or folder containing weights saved in safetensors
    files.

    For microscale schemes (NVFP4, MXFP4), fused weight sets (q/k/v, gate/up)
    are handled correctly even when split across shards. Each shard job receives
    a precomputed inverse_weights_map specifying exactly which tensors to load
    from which files — enabling true partial reads with no runtime discovery
    and no redundant tensor reads.

    :param model_stub: huggingface model hub or path to local weights files
    :param save_directory: directory to save quantized weights to
    :param scheme: weight quantization scheme or preset scheme name
    :param ignore: modules to ignore. Modules ending with "norm" are
        automatically ignored
    :param max_workers: number of worker threads to process files with
    :param device: gpu device to accelerate quantization with
    :param converter: optional converter to apply to the checkpoint to convert
        it to compressed-tensors format before running model-free PTQ
    """
    # validate arguments
    model_files = get_checkpoint_files(model_stub)
    scheme_name, scheme = validate_scheme(scheme)
    device = gpu_if_available(device)
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
    if is_microscale_scheme(scheme):
        jobs = _build_microscale_jobs(
            model_files, save_directory, scheme, ignore, device, converter
        )
    else:
        jobs = _build_standard_jobs(
            model_files, save_directory, scheme, ignore, device, converter
        )

    # 1. validate quantizable tensors — fail fast before long-running quantization
    validate_jobs = _build_validate_jobs(jobs)
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


def _build_standard_jobs(
    model_files: dict[str, str],
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: torch.device,
    converter: Converter | None,
    job_fn=None,
) -> list[tuple]:
    """Build one job per safetensors file using the given processing function."""
    if job_fn is None:
        job_fn = process_file
    jobs = []
    for file_path, resolved_path in model_files.items():
        if file_path.endswith("safetensors"):
            save_path = Path(save_directory) / file_path
            jobs.append(
                (job_fn, resolved_path, save_path, scheme, ignore, device, converter)
            )
    return jobs


def _build_microscale_jobs(
    model_files: dict[str, str],
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: torch.device,
    converter: Converter | None,
) -> list[tuple]:
    """
    Build microscale jobs with precomputed inverse_weights_map per shard.

    For each output shard, build_inverse_weights_map() determines exactly which
    tensors to load from which source files — including any fused partner tensors
    from other shards. This avoids runtime fused-partner discovery inside the
    process function and eliminates redundant tensor reads.

    Job tuple format:
        (process_file_microscale_scheme, inverse_weights_map, save_path,
         scheme, ignore, device, converter)
    """
    index_file = find_safetensors_index_file(model_files)

    if index_file is None:
        # Single-file model — no cross-shard fused weights possible,
        # Create inverse_weights_map dict format for process_file_microscale_scheme
        jobs = []
        for file_path, resolved_path in model_files.items():
            if file_path.endswith("safetensors"):
                save_path = Path(save_directory) / file_path
                # Wrap as inverse_weights_map: {source_file: None}
                # means load all tensors
                inverse_weights_map = {resolved_path: []}
                jobs.append(
                    (
                        process_file_microscale_scheme,
                        inverse_weights_map,
                        save_path,
                        scheme,
                        ignore,
                        device,
                        converter,
                    )
                )
        return jobs

    # Read weight map from safetensors.index.json
    with open(index_file, "r") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    jobs = []
    for shard_name, resolved_path in model_files.items():
        if not shard_name.endswith("safetensors"):
            continue

        save_path = Path(save_directory) / shard_name

        # Precompute exactly which tensors to load from which files for this shard,
        # including fused partner tensors that live in other shards
        inverse_weights_map = build_inverse_weights_map(
            shard_name=shard_name,
            weight_map=weight_map,
            model_files=model_files,
        )

        if len(inverse_weights_map) > 1:
            partner_shards = [s for s in inverse_weights_map if s != resolved_path]
            logger.info(
                f"{shard_name}: will fetch fused partners from "
                f"{[os.path.basename(s) for s in partner_shards]}"
            )

        jobs.append(
            (
                process_file_microscale_scheme,
                inverse_weights_map,
                save_path,
                scheme,
                ignore,
                device,
                converter,
            )
        )

    return jobs


def _build_validate_jobs(jobs: list[tuple]) -> list[tuple]:
    """
    Build validation jobs from processing jobs.

    Handles both job formats:
    - Standard/fallback: (proc_fn, file_path_str, save_path, scheme, ignore, device, \
        converter)
    - Microscale with index: (proc_fn, inverse_weights_map_dict, save_path, scheme, \
        ignore, device, converter)
    """
    validate_jobs = []
    for job in jobs:
        # job[0] is the processing function
        # Check if second element is a dict (microscale with index)
        # or string (standard/fallback)
        second_arg = job[1]

        if isinstance(second_arg, dict):
            # Microscale job with inverse_weights_map dict
            _, inverse_weights_map, save_path, scheme, ignore, device, converter = job
            # Use first source file path from inverse_weights_map for validation
            source_file = next(iter(inverse_weights_map.keys()))
            validate_jobs.append(
                (
                    validate_file,
                    source_file,
                    save_path,
                    scheme,
                    ignore,
                    device,
                    converter,
                    inverse_weights_map,
                )
            )
        else:
            # Standard job or microscale fallback: second_arg is file_path string
            _, file_path, save_path, scheme, ignore, device, converter = job
            validate_jobs.append(
                (
                    validate_file,
                    file_path,
                    save_path,
                    scheme,
                    ignore,
                    device,
                    converter,
                    None,
                )
            )
    return validate_jobs


def _get_all_tensor_names(file_path: str) -> list[str]:
    """Get all tensor names from a safetensors file without loading tensors."""
    from safetensors import safe_open

    with safe_open(file_path, framework="pt", device="cpu") as f:
        return list(f.keys())
