import os
from collections import defaultdict
from typing import Iterable

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils import match_quantizable_tensors
from safetensors import safe_open
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
    inverse_weights_map: dict[str, list[str] | None],
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: str | torch.device,
    converter: Converter | None = None,
):
    """
    Validate that each quantizable tensor in a safetensors file can be quantized.

    :param inverse_weights_map: mapping of source file path -> tensor names to validate
    :param save_path: save path of file with quantized weights
    :param scheme: quantization scheme to apply to tensors
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param device: device used to quantize and compress weights
    :param converter: optional converter to apply to the checkpoint,
        e.g. conversion of some layers from some format to compressed-tensors
    """
    tensors = _load_tensors_from_inverse_weights_map(inverse_weights_map, device)

    if converter is not None:
        converter.validate(tensors)

    for _, name in match_quantizable_tensors(tensors, ignore, scheme.targets):
        validate_weight_for_quantization(tensors[name], scheme, name)


def process_file(
    inverse_weights_map: dict[str, list[str] | None],
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: str | torch.device,
    converter: Converter | None = None,
) -> tuple[int, dict[str, str]]:
    """
    Quantize and compress tensors in a given safetensors file.

    :param inverse_weights_map: mapping of source file path -> tensor names.
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

    tensors = _load_tensors_from_inverse_weights_map(inverse_weights_map, device)

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
    inverse_weights_map: dict[str, list[str]],
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: str | torch.device,
    converter: Converter | None = None,
) -> tuple[int, dict[str, str]]:
    """
    Quantize and compress tensors for a single output shard using a microscale
    scheme (NVFP4, MXFP4).

    Accepts a precomputed inverse_weights_map that specifies exactly which tensors
    to load from which source files — including any fused partner tensors from
    other shards needed for global scale computation. This avoids runtime
    discovery of fused partners and redundant tensor reads.

    Partner tensors fetched from other shards are re-saved into this shard's
    output. The caller updates the safetensors index to reflect new locations.

    :param inverse_weights_map: mapping of resolved source file path ->
        list of tensor names to load from that file. Precomputed by
        build_microscale_inverse_weights_map() in the job-building phase.
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

    tensors = _load_tensors_from_inverse_weights_map(inverse_weights_map, device)

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


# TODO brian-dellabetta (#2491): move to compressed-tensors.utils.safetensors_load
def _load_tensors_from_inverse_weights_map(
    inverse_weights_map: dict[str, list[str] | None],
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """
    Given an inverse_weights_map, which is a dictionary of file name to list of
    tensor names, load up all listed tensor names

    :param inverse_weights_map: mapping of resolved source file path ->
        list of tensor names to load from that file. Precomputed by
        build_inverse_weights_map() in the job-building phase.
        If list is empty, all tensors are pulled
        Example: {"/path/shard0.safetensors": ["q_proj.weight"],
                  "/path/shard1.safetensors": ["k_proj.weight", "v_proj.weight"]}
    :param device: tensors will be loaded onto this device.

    :returns: mapping of tensor name to actual tensor loaded from safetensors file
        Example: {"q_proj.weight": torch.Tensor(...), "k_proj.weight: torch.Tensor(...)}
    """
    tensors: dict[str, torch.Tensor] = {}
    for source_file, tensor_names in inverse_weights_map.items():
        with safe_open(source_file, framework="pt", device=str(device)) as f:
            keys = f.keys()
            # if tensor_names is empty, pull all tensors
            if tensor_names is None or len(tensor_names) == 0:
                tensor_names = keys
            for tensor_name in tensor_names:
                if tensor_name not in keys:
                    raise ValueError(
                        f"Expected to find tensor {tensor_name} in "
                        f"{source_file}, but tensor was not found."
                    )
                tensors[tensor_name] = f.get_tensor(tensor_name)
    return tensors
