import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import torch
import tqdm
from compressed_tensors.quantization import QuantizationScheme
from loguru import logger
from safetensors.torch import load_file, save_file

from llmcompressor.entrypoints.weights_ptq.helpers import (
    gpu_if_available,
    is_match_name,
    validate_scheme,
)
from llmcompressor.entrypoints.weights_ptq.lifecycle import (
    calibrate_weights,
    compress_module,
    initialize_quantized_linear,
)
from llmcompressor.entrypoints.weights_ptq.model_utils import (
    get_checkpoint_files,
    is_weights_file,
)
from llmcompressor.entrypoints.weights_ptq.save_utils import (
    update_config,
    update_safetensors_index,
)

__all__ = ["ptq_weights"]


def ptq_weights(
    model_stub: str | os.PathLike,
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme | str,
    ignore: Optional[list[str]] = None,
    max_workers: int = 1,
    device: Optional[torch.device | str] = None,
):
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
    tensors = load_file(file_path)

    for name in list(tensors.keys()):
        if not is_match_name(name, ["re:.*weight$"], ignore):
            continue

        # 1. initialize module with qparams (on device)
        module = initialize_quantized_linear(tensors[name], scheme, device)

        # 2. calibrate weight qparams
        calibrate_weights(module)

        # 3. compress module using qparams
        compress_module(module)

        # 4. save compressed data (on cpu)
        del tensors[name]
        prefix = name.rsplit(".", 1)[0] + "."
        for key, value in module.state_dict(prefix=prefix).items():
            tensors[key] = value.to("cpu")

    save_file(tensors, save_path)
    total_size = sum(tensor.nbytes for tensor in tensors.values())
    weight_map = {key: os.path.basename(save_path) for key in tensors.keys()}
    return total_size, weight_map
