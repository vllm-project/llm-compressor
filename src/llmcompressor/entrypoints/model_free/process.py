import os
from collections import defaultdict
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils.match import _match_name
from safetensors.torch import load_file, save_file
from torch.nn import Module

from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_global_scale,
    calibrate_scale_zp,
    compress_module,
    initialize_quantized_linear,
)
from llmcompressor.entrypoints.model_free.microscale import (
    get_fused_names,
    is_microscale_scheme,
)

__all__ = ["process_file", "process_file_microscale_scheme"]

# Type alias for padded layers info: maps module name to original shape
PaddedLayersInfo = dict[str, dict[str, list[int]]]


def process_file(
    file_path: str | os.PathLike,
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: str | list[str],
    device: str | torch.device,
) -> tuple[int, dict[str, str], Optional[PaddedLayersInfo]]:
    """
    Quantize and compress tensors in a given safetensors file

    :param file_path: safetensors file to process
    :param save_path: save path of file with quantized weights
    :param scheme: quantization scheme to apply to tensors
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param device: device used to quantize and compress weights
    :return: Tuple of (total_size, weight_map, padded_layers) where padded_layers
        is a dict mapping module names to their original shapes before padding,
        or None if no layers were padded
    """
    assert not is_microscale_scheme(scheme), "Use `_process_file_microscale_scheme`"
    tensors = load_file(file_path)
    padded_layers: PaddedLayersInfo = {}

    for name in list(tensors.keys()):
        module_name, param_name = name.rsplit(".", 1)
        is_linear_weight = param_name == "weight" and not module_name.endswith("norm")
        is_ignored = any(_match_name(module_name, ign) for ign in ignore)
        if not is_linear_weight or is_ignored:
            continue

        # 1. initialize module with qparams (on device), pad if needed
        module, original_shape = initialize_quantized_linear(
            tensors[name], scheme, device
        )

        # Track original shape if padding was applied
        if original_shape is not None:
            padded_layers[module_name] = {
                "original_shape": list(original_shape),
            }

        # 2. calibrate weight qparams
        calibrate_scale_zp(module)

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
    return total_size, weight_map, padded_layers if padded_layers else None


def process_file_microscale_scheme(
    file_path: str | os.PathLike,
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: str | list[str],
    device: str | torch.device,
) -> tuple[int, dict[str, str], Optional[PaddedLayersInfo]]:
    """
    Quantize and compress tensors in a given safetensors file

    :param file_path: safetensors file to process
    :param save_path: save path of file with quantized weights
    :param scheme: quantization scheme to apply to tensors
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param device: device used to quantize and compress weights
    :return: Tuple of (total_size, weight_map, padded_layers) where padded_layers
        is a dict mapping module names to their original shapes before padding,
        or None if no layers were padded
    """
    assert is_microscale_scheme(scheme), "Use `_process_file` for non-microscale scheme"
    tensors = load_file(file_path)
    fused_sets, unmatched_sets = get_fused_names(tensors)
    assert len(unmatched_sets) <= 0  # should be caught by `validate_safetensors_index`

    fused_name_to_fused_index: dict[str, int]  # fused_name -> fused_index
    fused_modules: dict[int, dict[str, Module]]  # fused_index -> named_modules
    fused_original_shapes: dict[int, dict[str, tuple[int, int]]]  # fused original shapes

    fused_name_to_fused_index = {
        name: index
        for index, matched_set in enumerate(fused_sets)
        for name in matched_set.values()
    }
    fused_modules = defaultdict(dict)
    fused_original_shapes = defaultdict(dict)
    padded_layers: PaddedLayersInfo = {}

    for name in list(tensors.keys()):
        module_name, param_name = name.rsplit(".", 1)
        is_linear_weight = param_name == "weight" and not module_name.endswith("norm")
        is_ignored = any(_match_name(module_name, ign) for ign in ignore)
        if not is_linear_weight or is_ignored:
            continue

        # 1. initialize module with qparams (on device), pad if needed
        module, original_shape = initialize_quantized_linear(
            tensors[name], scheme, device
        )

        # 2. calibrate weight qparams. Delay scale/zp calibration for fused modules
        calibrate_global_scale(module)
        if name in fused_name_to_fused_index:
            fused_index = fused_name_to_fused_index[name]
            fused_modules[fused_index][name] = module
            # Track original shape for fused modules
            if original_shape is not None:
                fused_original_shapes[fused_index][module_name] = original_shape
            continue

        # Track original shape if padding was applied
        if original_shape is not None:
            padded_layers[module_name] = {
                "original_shape": list(original_shape),
            }

        calibrate_scale_zp(module)

        # 3. compress module using qparams
        compress_module(module)

        # 4. save compressed data (on cpu)
        del tensors[name]
        prefix = module_name + "."
        for key, value in module.state_dict(prefix=prefix).items():
            tensors[key] = value.to("cpu")

    # compress and save miscroscale fused modules
    for fused_index, named_modules in fused_modules.items():
        # 2.1. fuse global scales
        global_scales = [m.weight_global_scale for m in named_modules.values()]
        fused_global_scale = torch.min(torch.cat(global_scales, dim=0))

        for name, module in named_modules.items():
            module_name, param_name = name.rsplit(".", 1)
            module.weight_global_scale.data.copy_(fused_global_scale)

            # Track original shape if padding was applied for fused modules
            if fused_index in fused_original_shapes:
                if module_name in fused_original_shapes[fused_index]:
                    padded_layers[module_name] = {
                        "original_shape": list(
                            fused_original_shapes[fused_index][module_name]
                        ),
                    }

            # 2.2. finish calibration with fused global scales
            calibrate_scale_zp(module)

            # 3. compress module using miscroscale qparams
            compress_module(module)

            # 4. save compressed data (on cpu)
            del tensors[name]
            prefix = module_name + "."
            for key, value in module.state_dict(prefix=prefix).items():
                tensors[key] = value.to("cpu")

    save_file(tensors, save_path)
    total_size = sum(tensor.nbytes for tensor in tensors.values())
    weight_map = {key: os.path.basename(save_path) for key in tensors.keys()}
    return total_size, weight_map, padded_layers if padded_layers else None
