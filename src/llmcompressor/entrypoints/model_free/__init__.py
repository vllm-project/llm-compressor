import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import torch
import tqdm
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils.match import _match_name
from loguru import logger
from safetensors.torch import load_file, save_file

from llmcompressor.entrypoints.model_free.helpers import (
    gpu_if_available,
    validate_scheme,
)
from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_weights,
    compress_module,
    initialize_quantized_linear,
)
from llmcompressor.entrypoints.model_free.model_utils import (
    get_checkpoint_files,
    is_weights_file,
)
from llmcompressor.entrypoints.model_free.save_utils import (
    update_config,
    update_safetensors_index,
)

__all__ = ["model_free_ptq"]


def model_free_ptq(
    model_stub: str | os.PathLike,
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme | str,
    ignore: Optional[list[str]] = None,
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

    # 0. collect safetensors files, copy files
    jobs = []
    for file_path, resolved_path in model_files:
        save_path = Path(save_directory) / file_path

        if file_path.endswith("safetensors"):
            jobs.append(
                (_process_file, resolved_path, save_path, scheme, ignore, device)
            )

        else:
            if is_weights_file(file_path):
                logger.warning(f"Skipping weights file {file_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying {file_path} {save_path}")
            shutil.copyfile(resolved_path, save_path)

    # 1-4. quantize and compress weights
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(*job) for job in jobs]

        total_size = 0
        weight_map = dict()
        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Quantizing"
        ):
            _total_size, _weight_map = future.result()
            total_size += _total_size
            weight_map.update(_weight_map)

    # 5. update config and safetensors index
    update_config(save_directory, scheme_name, scheme, ignore)
    update_safetensors_index(save_directory, total_size, weight_map)


def _process_file(
    file_path: str | os.PathLike,
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: str | list[str],
    device: str | torch.device,
) -> tuple[int, dict[str, str]]:
    """
    Quantize and compress tensors in a given safetensors file

    :param file_path: safetensors file to process
    :param save_path: save path of file with quantized weights
    :param scheme: quantization scheme to apply to tensors
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param device: device used to quantize and compress weights
    """
    tensors = load_file(file_path)

    for name in list(tensors.keys()):
        module_name, param_name = name.rsplit(".", 1)
        is_linear_weight = param_name == "weight" and not module_name.endswith("norm")
        is_ignored = any(_match_name(module_name, ign) for ign in ignore)
        if not is_linear_weight or is_ignored:
            continue

        # 1. initialize module with qparams (on device)
        module = initialize_quantized_linear(tensors[name], scheme, device)

        # 2. calibrate weight qparams
        calibrate_weights(module)

        # 3. compress module using qparams
        compress_module(module)

        # 4. save compressed data (on cpu)
        del tensors[name]
        prefix = module_name + "."
        for key, value in module.state_dict(prefix=prefix).items():
            tensors[key] = value.to("cpu")

    save_file(tensors, save_path)
    total_size = sum(tensor.nbytes for tensor in tensors.values())
    weight_map = {key: os.path.basename(save_path) for key in tensors.keys()}
    return total_size, weight_map
