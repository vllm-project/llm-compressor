import os
from collections import defaultdict
from typing import Iterable

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils import match_quantizable_tensors
from compressed_tensors.utils.match import match_name
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.nn import Module

from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_global_scale,
    calibrate_scale_zp,
    initialize_quantized_linear,
    validate_weight_for_quantization,
)
from llmcompressor.entrypoints.model_free.microscale import (
    DEFAULT_FUSED_MAPPINGS,
    get_fused_names,
    is_microscale_scheme,
)

__all__ = [
    "validate_file",
    "process_file",
    "process_file_microscale_scheme",
]


def validate_file(
    file_path: str | os.PathLike,
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: str | torch.device,
    converter: Converter | None = None,
    weights_map: dict[str, str] | None = None,
):
    """
    Validate that each quantizable tensor in a safetensors file can be quantized.

    :param file_path: safetensors file to validate
    :param scheme: quantization scheme to apply to tensors
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param converter: optional converter to apply to the checkpoint,
        e.g. conversion of some layers from some format to compressed-tensors
    """
    tensors = load_file(file_path)

    if converter is not None:
        converter.validate(tensors)

    for _, name in match_quantizable_tensors(tensors, ignore, scheme.targets):
        validate_weight_for_quantization(tensors[name], scheme, name)


def process_file(
    file_path: str | os.PathLike,
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: str | torch.device,
    converter: Converter | None = None,
) -> tuple[int, dict[str, str]]:
    """
    Quantize and compress tensors in a given safetensors file

    :param file_path: safetensors file to process
    :param save_path: save path of file with quantized weights
    :param scheme: quantization scheme to apply to tensors
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param device: device used to quantize and compress weights
    :param converter: optional converter to apply to the checkpoint,
        e.g. conversion of some layers from some format to compressed-tensors
    """
    assert not is_microscale_scheme(scheme), "Use `process_file_microscale_scheme`"
    tensors = load_file(file_path)

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
    file_path: str | os.PathLike,
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: Iterable[str],
    device: str | torch.device,
    converter: Converter | None = None,
    tensor_file_index: dict[str, str] | None = None,
) -> tuple[int, dict[str, str]]:
    """
    Quantize and compress tensors in a given safetensors file using a microscale
    scheme (NVFP4, MXFP4).

    When fused weight sets (q/k/v, gate/up) are split across shards, uses
    tensor_file_index to perform true partial reads (via safe_open) of only
    the fused partner tensors needed for global scale computation. Each process
    writes only its own output shard — no cross-process coordination required.

    :param file_path: safetensors file to process
    :param save_path: save path of file with quantized weights
    :param scheme: microscale quantization scheme (NVFP4, MXFP4)
    :param ignore: modules to ignore. Modules ending with "norm" are automatically
        ignored
    :param device: device used to quantize and compress weights
    :param converter: optional converter to apply to the checkpoint,
        e.g. conversion of some layers from some format to compressed-tensors
    :param tensor_file_index: optional mapping of tensor name -> source file path,
        built from safetensors.index.json. When provided, enables true partial reads of fused
        partner tensors from other shards for correct global scale fusion.
    """
    assert is_microscale_scheme(scheme), "Use `process_file` for non-microscale scheme"

    # Load this shard's tensors and identify native tensors in one pass
    tensors = load_file(file_path)
    native_tensor_names: set[str] = set(tensors.keys())

    if converter is not None:
        converter.process(tensors)

    # Fetch any fused partner tensors that live in other shards
    if tensor_file_index is not None:
        tensors = _fetch_fused_partners(tensors, tensor_file_index, file_path)

    fused_sets, unmatched_sets = get_fused_names(list(tensors.keys()))

    if len(unmatched_sets) > 0 and tensor_file_index is None:
        raise NotImplementedError(
            "When using a microscale scheme (NVFP4, MXFP4), global scales "
            "will be fused. Current implementation requires that all fused "
            "modules (attention and mlp) be stored in the same file. "
            f"However, this file has an unmatched set of fused weights: "
            f"{unmatched_sets}\n\n"
            "Please pass a tensor_file_index built from the model's index.json "
            "to enable cross-shard fused weight processing."
        )

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

        # 4. save compressed data — only for native tensors
        if name in native_tensor_names:
            del tensors[name]
            prefix = module_name + "."
            for key, value in module.state_dict(prefix=prefix).items():
                tensors[key] = value.to("cpu")

    # Compress and save microscale fused modules (with fused global scales)
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

            # 4. save compressed data — only for native tensors
            if name in native_tensor_names:
                del tensors[name]
                prefix = module_name + "."
                for key, value in module.state_dict(prefix=prefix).items():
                    tensors[key] = value.to("cpu")

    # Remove any partner tensors fetched for scale computation only
    output_tensors = {
        k: v for k, v in tensors.items() if _belongs_to_shard(k, native_tensor_names)
    }

    save_file(output_tensors, save_path)
    total_size = sum(tensor.nbytes for tensor in output_tensors.values())
    weight_map = {key: os.path.basename(save_path) for key in output_tensors.keys()}
    return total_size, weight_map


def _fetch_fused_partners(
    tensors: dict[str, torch.Tensor],
    weights_map: dict[str, str],
    own_file_path: str | os.PathLike,
) -> dict[str, torch.Tensor]:
    """
    For any tensor in this shard that is part of a fused set, fetch the
    partner tensors from their source shards via true partial reads using
    safe_open. Only the specific tensor names needed are read — not entire
    files — keeping I/O proportional to the number of cross-shard fused
    weights, not shard size.

    :param tensors: tensors already loaded from this shard
    :param tensor_file_index: mapping of tensor name -> source file path
    :param own_file_path: resolved path of the current shard (to skip self-reads)
    :return: tensors dict augmented with any fetched fused partners
    """

    from llmcompressor.entrypoints.model_free.microscale import (
        get_fused_names,
    )

    own_file_path = str(os.path.abspath(own_file_path))

    # Find which tensors in this shard are members of incomplete fused sets
    _, unmatched_sets = get_fused_names(list(tensors.keys()))
    if not unmatched_sets:
        return tensors  # All fused sets already complete in this shard

    # For each unmatched set, derive the shared layer prefix from present members,
    # then scan weights_map once to find missing partners at that prefix.
    tensors_to_fetch: dict[str, str] = {}  # tensor_name -> source_file
    all_patterns = [p for mapping in DEFAULT_FUSED_MAPPINGS for p in mapping]

    for unmatched in unmatched_sets:
        present_names = {v for v in unmatched.values() if v is not None}
        layer_prefixes = {name.rsplit(".", 2)[0] for name in present_names}

        for candidate_name, candidate_file in weights_map.items():
            if (
                candidate_name in tensors
                or candidate_name in tensors_to_fetch
                or str(os.path.abspath(candidate_file)) == own_file_path
            ):
                continue
            candidate_prefix = candidate_name.rsplit(".", 2)[0]
            if candidate_prefix not in layer_prefixes:
                continue
            if any(match_name(candidate_name, p) for p in all_patterns):
                tensors_to_fetch[candidate_name] = candidate_file

    if not tensors_to_fetch:
        return tensors

    # Group fetches by source file to minimize file opens
    by_file: dict[str, list[str]] = defaultdict(list)
    for tensor_name, source_file in tensors_to_fetch.items():
        by_file[source_file].append(tensor_name)

    # True partial read: use safe_open to load only the specific tensors needed,
    # not the entire shard. This keeps memory proportional to the number of
    # cross-shard fused weights, not file size.
    for source_file, tensor_names in by_file.items():
        with safe_open(source_file, framework="pt", device="cpu") as f:
            for name in tensor_names:
                if name in f.keys():
                    tensors[name] = f.get_tensor(name)

    return tensors


def _belongs_to_shard(
    tensor_name: str,
    native_tensor_names: set[str],
) -> bool:
    """
    Returns True if a tensor key belongs to this shard's output.
    Handles the case where compression expands a weight key into
    multiple keys (e.g. weight -> weight_scale, weight_zero_point).
    """
    if tensor_name in native_tensor_names:
        return True
    for native in native_tensor_names:
        if tensor_name.startswith(native.rsplit(".", 1)[0] + "."):
            return True
    return False
