import os
from collections import defaultdict
from typing import Iterable

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils import match_quantizable_tensors
from compressed_tensors.utils.safetensors_load import (
    InverseWeightMap,
    load_tensors_from_inverse_weight_map,
)
from loguru import logger
from safetensors.torch import save_file
from torch.nn import Module

from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_global_scale,
    calibrate_scale_zp,
    initialize_quantized_linear,
    validate_weight_for_quantization,
)
from llmcompressor.entrypoints.model_free.microscale import (
    get_fused_names,
    is_microscale_scheme,
)

__all__ = [
    "validate_file",
    "process_file",
    "process_file_microscale_scheme",
]


def validate_file(
    inverse_weight_map: InverseWeightMap,
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: str | torch.device,
    converter: Converter | None = None,
):
    """
    Validate that each quantizable tensor in a safetensors file can be quantized.

    :param inverse_weight_map: mapping of source file path -> tensor names to validate
    :param save_path: save path of file with quantized weights
    :param scheme: quantization scheme to apply to tensors
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param device: device used to quantize and compress weights
    :param converter: optional converter to apply to the checkpoint,
        e.g. conversion of some layers from some format to compressed-tensors
    """
    tensors = load_tensors_from_inverse_weight_map(inverse_weight_map, device)

    if converter is not None:
        converter.validate(tensors)

    for _, name in match_quantizable_tensors(tensors, ignore, scheme.targets):
        validate_weight_for_quantization(tensors[name], scheme, name)


def process_file(
    inverse_weight_map: InverseWeightMap,
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: str | torch.device,
    converter: Converter | None = None,
) -> tuple[int, dict[str, str]]:
    """
    Quantize and compress tensors in a given safetensors file.

    :param inverse_weight_map: mapping of source file path -> tensor names.
        For standard mode: {{resolved_path: None}} means load all tensors to process
    :param save_path: save path of file with quantized weights
    :param scheme: quantization scheme to apply to tensors
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param device: device used to quantize and compress weights
    :param converter: optional converter to apply to the checkpoint,
        e.g. conversion of some layers from some format to compressed-tensors
    """
    assert not is_microscale_scheme(scheme), "Use `process_file_microscale_scheme`"

    tensors = load_tensors_from_inverse_weight_map(inverse_weight_map, device)

    tensors = split_fused_moe_experts(tensors)

    if converter is not None:
        converter.process(tensors)

    for module_name, name in match_quantizable_tensors(tensors, ignore, scheme.targets):
        validate_weight_for_quantization(tensors[name], scheme, name)

        # 1. initialize module with qparams (on device)
        module = initialize_quantized_linear(tensors[name], scheme, device)

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
    return total_size, weight_map


def process_file_microscale_scheme(
    inverse_weight_map: InverseWeightMap,
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: str | torch.device,
    converter: Converter | None = None,
) -> tuple[int, dict[str, str]]:
    """
    Quantize and compress tensors for a single output shard using a microscale
    scheme (NVFP4, MXFP4).

    Accepts a precomputed inverse_weight_map that specifies exactly which tensors
    to load from which source files — including any fused partner tensors from
    other shards needed for global scale computation. This avoids runtime
    discovery of fused partners and redundant tensor reads.

    Partner tensors fetched from other shards are re-saved into this shard's
    output. The caller updates the safetensors index to reflect new locations.

    :param inverse_weight_map: mapping of resolved source file path ->
        list of tensor names to load from that file.
        Example: {"/path/shard0.safetensors": ["q_proj.weight"],
                  "/path/shard1.safetensors": ["k_proj.weight", "v_proj.weight"]}
    :param save_path: output path for this shard's compressed weights
    :param scheme: microscale quantization scheme (NVFP4, MXFP4)
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param device: device used to quantize and compress weights
    :param converter: optional converter to apply to the checkpoint,
        e.g. conversion of some layers from some format to compressed-tensors
    """
    assert is_microscale_scheme(scheme), "Use `process_file` for non-microscale scheme"

    tensors = load_tensors_from_inverse_weight_map(inverse_weight_map, device)

    tensors = split_fused_moe_experts(tensors)

    if converter is not None:
        converter.process(tensors)

    # Get fused sets. Non-primary shards may have incomplete sets (k/v without q)
    # since only the primary-owning shard fetches partners — this is correct.
    fused_sets, _ = get_fused_names(list(tensors.keys()))

    fused_name_to_fused_index: dict[str, int] = {
        name: index
        for index, matched_set in enumerate(fused_sets)
        for name in matched_set.values()
        if name is not None
    }
    fused_modules: dict[int, dict[str, Module]] = defaultdict(dict)

    for module_name, name in match_quantizable_tensors(tensors, ignore, scheme.targets):
        validate_weight_for_quantization(tensors[name], scheme, name)

        # 1. initialize module with qparams (on device)
        module = initialize_quantized_linear(tensors[name], scheme, device)

        # 2. calibrate global scale; delay scale/zp for fused modules
        calibrate_global_scale(module)
        if name in fused_name_to_fused_index:
            fused_index = fused_name_to_fused_index[name]
            fused_modules[fused_index][name] = module
            continue

        calibrate_scale_zp(module)

        # 3. compress module using qparams
        compress_module(module)

        # 4. save compressed data (on cpu)
        del tensors[name]
        prefix = module_name + "."
        for key, value in module.state_dict(prefix=prefix).items():
            tensors[key] = value.to("cpu")

    # Compress fused modules with shared global scale
    for named_modules in fused_modules.values():
        # 2.1. compute fused global scale across all members of the fused set
        global_scales = [m.weight_global_scale for m in named_modules.values()]
        fused_global_scale = torch.min(torch.cat(global_scales, dim=0))

        for name, module in named_modules.items():
            module_name, _ = name.rsplit(".", 1)
            module.weight_global_scale.data.copy_(fused_global_scale)

            # 2.2. finish calibration with fused global scale
            calibrate_scale_zp(module)

            # 3. compress module using microscale qparams
            compress_module(module)

            # 4. save compressed data (on cpu)
            del tensors[name]
            prefix = module_name + "."
            for key, value in module.state_dict(prefix=prefix).items():
                tensors[key] = value.to("cpu")

    # Save ALL tensors to this shard's output — including partner tensors fetched
    # from other shards. Partners are re-saved here so future runs don't need to
    # re-fetch them. The caller updates the safetensors index to reflect new locations.
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    save_file(tensors, save_path)
    total_size = sum(t.nbytes for t in tensors.values())
    weight_map = {key: os.path.basename(save_path) for key in tensors.keys()}
    return total_size, weight_map


def split_fused_moe_experts(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Find fused MoE experts (with gate_up_proj/down_proj).
    Split them from 3D tensors into individual 2D expert tensors.

    Args:
        tensors: Dictionary of loaded tensors from safetensors file

    Returns:
        split_tensors: New dictionary with split expert weights
    """
    split_tensors = {}

    params_to_split = {
        # If a 3D gate_up_proj layer is found, split it into a
        # 2D gate_proj and up_proj layer for each expert
        "gate_up_proj": ["gate_proj", "up_proj"],
        # If a 3D down_proj layer is found, split it into a
        # 2D down_proj layer for each expert
        "down_proj": ["down_proj"],
    }

    for name, tensor in tensors.items():
        keys_to_split = [key for key in params_to_split if key in name]
        if len(keys_to_split) >= 2:
            raise ValueError(f"Found multiple keys matching {name}: {keys_to_split}")

        elif len(keys_to_split) == 1 and tensor.ndim == 3:
            unsplit_name = keys_to_split[0]
            split_names = params_to_split[unsplit_name]

            # Get number of experts
            num_experts = tensor.shape[0]

            if tensor.shape[1] % len(split_names) != 0:
                raise ValueError(
                    f"{unsplit_name} expects a second dimension divisible by "
                    f"{len(split_names)} but got shape: {tensor.shape}"
                )

            # Split into experts
            intermediate_size = tensor.shape[1] // len(split_names)
            for expert_idx in range(num_experts):
                expert_tensor = tensor[expert_idx]
                # Split into layers
                split_layers = expert_tensor.split(intermediate_size, dim=0)
                for split_name, split_layer in zip(split_names, split_layers):
                    key = name.replace(unsplit_name, f"{expert_idx}.{split_name}")
                    split_tensors[key] = split_layer

            logger.info(f"Split {name} into {num_experts} experts")

        else:
            # Non-MoE or non-3D tensors, keep as is
            split_tensors[name] = tensor

    return split_tensors
