import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import torch
import tqdm
from compressed_tensors.quantization import QuantizationScheme, preset_name_to_scheme
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.match import _match_name
from loguru import logger
from safetensors.torch import load_file, save_file

from .lifecycle import calibrate_weights, compress_module, initialize_quantized_linear
from .save_utils import update_config, update_safetensors_index

__all__ = ["weights_ptq"]


def weights_ptq(
    model_path: str,
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme | str,
    ignore: Optional[list[str]] = None,
    max_workers: int = 1,
    device: Optional[torch.device | str] = "cuda:0",
):
    # validate arguments
    scheme_name, scheme = _validate_scheme(scheme)
    os.makedirs(save_directory, exist_ok=True)

    # 0. collect safetensors files, copy files
    jobs = []
    for file_name in os.listdir(model_path):
        file_path = os.path.join(model_path, file_name)
        save_path = os.path.join(model_path, file_name)

        if file_name.endswith("safetensors"):
            jobs.append((_process_file, file_path, save_path, scheme, ignore, device))

        elif os.path.isdir(file_path):
            shutil.copytree(file_path, save_path, dirs_exist_ok=True)
        else:
            shutil.copyfile(file_path, save_path)

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
        if not _is_match_name(name, ["re:.*"], ignore):
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


def _validate_scheme(scheme: QuantizationScheme) -> tuple[str, QuantizationScheme]:
    # treat strings as preset schemes
    if isinstance(scheme, str):
        scheme_name, scheme = scheme, preset_name_to_scheme(scheme, [])
    else:
        scheme_name = "config_group_0"

    # weight quantization must be provided
    if scheme.weights is None:
        raise ValueError()

    # input quantization must be dynamic
    input_dynamic_attr = "input_activations.input_activations.dynamic"
    if getattr_chain(scheme, input_dynamic_attr, True) is not True:
        raise ValueError()

    # output quantization must be dynamic
    output_dynamic_attr = "input_activations.input_activations.dynamic"
    if getattr_chain(scheme, output_dynamic_attr, True) is not True:
        raise ValueError()

    # override with static observers
    if scheme.weights.observer in ("minmax", "mse"):
        new_observer = f"static_{scheme.weights.observer}"
        logger.warning(
            f"Scheme uses {scheme.weights.observer} weight observer. "
            f"Using {new_observer} instead"
        )
        scheme.weights.observer = new_observer

    # target all modules; filter by ignore list
    # technically this should be "re:.*", but vllm's
    # ct moe layer has a hard coded check for "Linear"
    scheme.targets = ["Linear"]
    return scheme_name, scheme


def _is_match_name(
    name: str, targets: list[str], ignore: Optional[str | list[str]] = None
) -> bool:
    targets = targets if isinstance(targets, list) else [targets]
    ignore = ignore if isinstance(ignore, list) else [ignore]

    matches_target = any(_match_name(name, target) for target in targets)
    matches_ignore = any(_match_name(name, ign) for ign in ignore)

    return matches_target and not matches_ignore


if __name__ == "__main__":
    weights_ptq(
        "Llama-3.2-1B-Instruct",
        "testing_save",
        scheme="FP8_BLOCK",
        ignore=[
            "model.embed_tokens.weight",
            "re:.*norm.weight$",
        ],
    )
