from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationScheme, preset_name_to_scheme
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.match import _match_name
from loguru import logger

__all__ = ["validate_scheme", "gpu_if_available", "is_match_name"]


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


def is_match_name(
    name: str, targets: list[str], ignore: Optional[str | list[str]] = None
) -> bool:
    targets = targets if isinstance(targets, list) else [targets]
    ignore = ignore if isinstance(ignore, list) else [ignore]

    matches_target = any(_match_name(name, target) for target in targets)
    matches_ignore = any(_match_name(name, ign) for ign in ignore)

    return matches_target and not matches_ignore
