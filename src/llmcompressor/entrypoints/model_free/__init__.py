import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import torch
import tqdm
from compressed_tensors.quantization import QuantizationScheme
from loguru import logger

from llmcompressor.entrypoints.model_free.helpers import gpu_if_available
from llmcompressor.entrypoints.model_free.microscale import (
    is_microscale_scheme,
)
from llmcompressor.entrypoints.model_free.model_utils import (
    get_checkpoint_files,
    is_weights_file,
)
from llmcompressor.entrypoints.model_free.process import (
    process_file,
    process_file_microscale_scheme,
    validate_file,
)
from llmcompressor.entrypoints.model_free.save_utils import (
    update_config,
    update_safetensors_index,
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
):
    """
    Quantize a model without the need for a model definition. This function operates on
    a model stub or folder containing weights saved in safetensors files

    :param model_stub: huggingface model hub or path to local weights files
    :param scheme: weight quantization scheme or preset scheme name
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param max_workers: number of worker threads to process files with
    :param device: gpu device to accelerate quantization with
    """
    # validate arguments
    model_files = get_checkpoint_files(model_stub)
    scheme_name, scheme = validate_scheme(scheme)
    device = gpu_if_available(device)
    validate_safetensors_index(model_files, scheme)

    # 0. collect safetensors files, copy files
    jobs = []
    job_fn = (
        process_file
        if not is_microscale_scheme(scheme)
        else process_file_microscale_scheme
    )
    for file_path, resolved_path in model_files.items():
        save_path = Path(save_directory) / file_path

        if file_path.endswith("safetensors"):
            jobs.append((job_fn, resolved_path, save_path, scheme, ignore, device))

        else:
            if is_weights_file(file_path):
                logger.warning(f"Skip processing for weights file {file_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying {file_path} {save_path}")
            shutil.copyfile(resolved_path, save_path)

    with ThreadPoolExecutor(max_workers) as executor:
        # 1. validate quantizable tensors fail fast before long-running quantization
        futures = [executor.submit(validate_file, *job[1:]) for job in jobs]
        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Validating"
        ):
            future.result()

        # 2-5. quantize and compress weights
        total_size = 0
        weight_map = dict()
        futures = [executor.submit(*job) for job in jobs]
        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Quantizing"
        ):
            _total_size, _weight_map = future.result()
            total_size += _total_size
            weight_map.update(_weight_map)

    # 5. update config and safetensors index
    update_config(save_directory, scheme_name, scheme, ignore)
    update_safetensors_index(save_directory, total_size, weight_map)
