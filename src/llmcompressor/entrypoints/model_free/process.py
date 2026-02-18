import os
from collections import defaultdict

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


def process_file(
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
    assert not is_microscale_scheme(scheme), "Use `_process_file_microscale_scheme`"
    tensors = load_file(file_path)

    for name in list(tensors.keys()):
        module_name, param_name = name.rsplit(".", 1)

        # rename params from modelopt to CT convention
        # modelopt's nvfp4-quantized layers, found by inspection
        # - model.layers.0.mlp.down_proj.weight
        # - model.layers.0.mlp.gate_proj.weight
        # - model.layers.0.mlp.up_proj.weight
        # - model.layers.3.mlp.shared_experts.down_proj.weight
        # - model.layers.3.mlp.shared_experts.gate_proj.weight
        # - model.layers.3.mlp.shared_experts.up_proj.weight
        # - model.layers.3.mlp.experts.0.down_proj.weight
        # - model.layers.3.mlp.experts.0.gate_proj.weight
        # - model.layers.3.mlp.experts.0.up_proj.weight
        if _match_name(module_name, "re:.*mlp.*\.(gate|up|down)_proj$"):
            match param_name:
                # input_scale -> input_global_scale F32
                case "input_scale":
                    # convert modelopt input_scale x -> 1/x
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1070-L1073
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1134
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L190
                    tensors[f"{module_name}.input_global_scale"] = 1 / tensors[name]
                    del tensors[name]
                # weight -> weight_packed U8
                case "weight":
                    # TODO reverse packing order(?)
                    tensors[f"{module_name}.weight_packed"] = tensors[name]
                    del tensors[name]
                # weight_scale -> weight_scale F8_E4M3
                case "weight_scale":
                    pass
                # weight_scale_2 -> weight_global_scale F32
                case "weight_scale_2":
                    # convert modelopt weight_scale_2 x -> 1/x
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1066-L1068
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L163-L166
                    tensors[f"{module_name}.weight_global_scale"] = 1 / tensors[name]
                    del tensors[name]
                case _:
                    print(f"Hit unexpected tensor {name}")

        is_linear_weight = param_name == "weight" and not module_name.endswith("norm")
        is_targeted = (is_linear_weight and "Linear" in scheme.targets) or any(
            _match_name(module_name, target) for target in scheme.targets
        )
        is_ignored = any(_match_name(module_name, ign) for ign in ignore)
        if is_ignored or not is_targeted:
            continue

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
    assert is_microscale_scheme(scheme), "Use `_process_file` for non-microscale scheme"
    tensors = load_file(file_path)
    fused_sets, unmatched_sets = get_fused_names(tensors)
    assert len(unmatched_sets) <= 0  # should be caught by `validate_safetensors_index`

    fused_name_to_fused_index: dict[str, int]  # fused_name -> fused_index
    fused_modules: dict[int, dict[str, Module]]  # fused_index -> named_modules

    fused_name_to_fused_index = {
        name: index
        for index, matched_set in enumerate(fused_sets)
        for name in matched_set.values()
    }
    fused_modules = defaultdict(dict)

    for name in list(tensors.keys()):
        module_name, param_name = name.rsplit(".", 1)
        is_linear_weight = param_name == "weight" and not module_name.endswith("norm")
        is_targeted = (is_linear_weight and "Linear" in scheme.targets) or any(
            _match_name(module_name, target) for target in scheme.targets
        )
        is_ignored = any(_match_name(module_name, ign) for ign in ignore)
        if is_ignored or not is_targeted:
            continue

        # 1. initialize module with qparams (on device)
        module = initialize_quantized_linear(tensors[name], scheme, device)

        # 2. calibrate weight qparams. Delay scale/zp calibration for fused modules
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

    # compress and save miscroscale fused modules
    for named_modules in fused_modules.values():
        # 2.1. fuse global scales
        global_scales = [m.weight_global_scale for m in named_modules.values()]
        fused_global_scale = torch.min(torch.cat(global_scales, dim=0))

        for name, module in named_modules.items():
            module_name, param_name = name.rsplit(".", 1)
            module.weight_global_scale.data.copy_(fused_global_scale)

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
    return total_size, weight_map
