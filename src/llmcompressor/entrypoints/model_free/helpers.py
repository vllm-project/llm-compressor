import json

import torch
from compressed_tensors.quantization import (
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.utils import getattr_chain
from loguru import logger

from .microscale import get_fused_names, is_microscale_scheme

__all__ = ["validate_scheme", "gpu_if_available"]


def validate_scheme(scheme: QuantizationScheme) -> tuple[str, QuantizationScheme]:
    # treat strings as preset schemes
    if isinstance(scheme, str):
        scheme_name, scheme = scheme, preset_name_to_scheme(scheme, [])
    else:
        scheme_name = "config_group_0"

    # weight quantization must be provided
    if scheme.weights is None:
        raise ValueError(
            "Must provide a weights quanitization scheme to perform weights-only PTQ"
        )

    # activation quantization must be dynamic
    input_dynamic = getattr_chain(scheme, "input_activations.dynamic", True)
    output_dynamic = getattr_chain(scheme, "output_activations.dynamic", True)
    if input_dynamic is not True or output_dynamic is not True:
        raise ValueError(
            "Model Free PTQ cannot calibrate activations. "
            "Please use `oneshot` instead."
        )

    # override with static observers
    # Remove after https://github.com/vllm-project/compressed-tensors/pull/489
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


def validate_safetensors_index(
    model_files: list[tuple[str, str]], scheme: QuantizationScheme
):
    resolved_paths = [
        resolved_path
        for file_path, resolved_path in model_files
        if file_path.endswith("safetensors.index.json")
    ]
    if len(resolved_paths) <= 0:
        return
    resolved_path = resolved_paths[0]

    if is_microscale_scheme(scheme):
        with open(resolved_path, "r") as file:
            weight_map: dict[str, str] = json.load(file)["weight_map"]

        fused_names = get_fused_names(weight_map)
        for submodule_names in fused_names.values():
            file_names = [weight_map[name] for name in submodule_names]
            if not all(file_name == file_names[0] for file_name in file_names):
                raise NotImplementedError(
                    "When using a microscale scheme (NVFP4, MXFP4), global scales "
                    "will be fused. Current implmentation requires that all fused "
                    "modules (attention and non-moe mlp) be stored in the same file. "
                    f"Instead, got {submodule_names}\n\n {file_names}"
                )


def gpu_if_available(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    elif torch.cuda.is_available():
        return torch.device("cuda:0")

    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")

    else:
        logger.warning("CUDA/XPU is not available! Compressing model on CPU instead")
        return torch.device("cpu")
