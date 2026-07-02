from collections.abc import Iterable

import torch
from compressed_tensors.modeling import KV_CACHE_ATTR
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStatus,
    is_attention_module,
)
from compressed_tensors.utils import match_named_modules
from torch.nn import Embedding, Linear, Module

__all__ = [
    "collect_calibration_modules",
    "format_calibration_error",
    "get_unvalidated_calibration_observers",
    "get_invalid_static_kv_cache_scales",
]

_OBSERVED_INPUT_DYNAMICS = (False, None, DynamicType.LOCAL, "local")
_STATIC_DYNAMICS = (False, None)
_SUPPORTED_CALIBRATION_MODULES = (Linear, Embedding)


def collect_calibration_modules(
    model: Module,
    targets: set[str],
    ignore: list[str],
    modules: Iterable[Module] | None = None,
) -> list[tuple[str, Module]]:
    named_modules = list(match_named_modules(model, targets, ignore))
    if modules is None:
        return named_modules

    module_ids = _module_tree_ids(modules)
    return [
        (name, module) for name, module in named_modules if id(module) in module_ids
    ]


def get_unvalidated_calibration_observers(
    named_modules: list[tuple[str, Module]],
    kv_cache_scheme: QuantizationArgs | None,
) -> list[str]:
    unobserved = []
    for module_name, module in named_modules:
        for base_name in _expected_observer_base_names(module, kv_cache_scheme):
            observer = getattr(module, f"{base_name}_observer", None)
            if _observer_num_observations(observer) <= 0:
                unobserved.append(
                    format_module_attr(module_name, f"{base_name}_observer")
                )

    return unobserved


def get_invalid_static_kv_cache_scales(
    named_modules: list[tuple[str, Module]],
    kv_cache_scheme: QuantizationArgs | None,
    unobserved_observers: list[str],
) -> list[str]:
    if _skip_static_kv_cache_validation(kv_cache_scheme):
        return []

    unobserved_scales = {
        observer_name.removesuffix("_observer") + "_scale"
        for observer_name in unobserved_observers
    }
    invalid_scales = []
    for module_name, module in named_modules:
        if _module_is_frozen(module) or not _is_static_kv_cache_module(
            module, kv_cache_scheme
        ):
            continue

        for base_name in ("k", "v"):
            param_name = f"{base_name}_scale"
            full_param_name = format_module_attr(module_name, param_name)
            if full_param_name in unobserved_scales:
                continue

            invalid_reason = _get_invalid_scale_reason(
                getattr(module, param_name, None)
            )
            if invalid_reason is not None:
                invalid_scales.append(f"{full_param_name} ({invalid_reason})")

    return invalid_scales


def format_calibration_error(
    unobserved_observers: list[str], invalid_scales: list[str]
) -> str:
    details = []
    if unobserved_observers:
        details.append(
            "missing or unobserved calibration observers: "
            f"{', '.join(unobserved_observers)}"
        )
    if invalid_scales:
        details.append(f"invalid static KV-cache scales: {', '.join(invalid_scales)}")

    return (
        "Quantization calibration failed. Calibration observers must be invoked "
        "before qparams are frozen, and static KV-cache scales must be finite "
        f"positive values. Found {'; '.join(details)}. This usually means one "
        "or more targeted modules were not executed during calibration."
    )


def format_module_attr(module_name: str, attr_name: str) -> str:
    return f"{module_name}.{attr_name}" if module_name else attr_name


def _module_tree_ids(modules: Iterable[Module]) -> set[int]:
    module_ids = set()
    for module in modules:
        module_ids.update(id(child) for child in module.modules())
    return module_ids


def _expected_observer_base_names(
    module: Module, kv_cache_scheme: QuantizationArgs | None
) -> Iterable[str]:
    scheme = getattr(module, "quantization_scheme", None)
    if scheme is None or _module_is_frozen(module):
        return ()

    if _is_static_kv_cache_module(module, kv_cache_scheme):
        return ("k", "v")

    if is_attention_module(module) or not isinstance(
        module, _SUPPORTED_CALIBRATION_MODULES
    ):
        return ()

    observer_names = []
    if _observer_enabled(getattr(scheme, "input_activations", None), is_input=True):
        observer_names.append("input")
    if getattr(scheme, "weights", None) is not None:
        observer_names.append("weight")
    if _observer_enabled(getattr(scheme, "output_activations", None), is_input=False):
        observer_names.append("output")

    return tuple(observer_names)


def _observer_enabled(args: QuantizationArgs | None, is_input: bool) -> bool:
    if args is None:
        return False

    dynamic = getattr(args, "dynamic", None)
    if input:
        return dynamic in _OBSERVED_INPUT_DYNAMICS

    return dynamic in _STATIC_DYNAMICS


def _is_static_kv_cache_module(
    module: Module, kv_cache_scheme: QuantizationArgs | None
) -> bool:
    if _skip_static_kv_cache_validation(kv_cache_scheme):
        return False

    return (
        getattr(module, "quantization_scheme", None) is not None
        and is_attention_module(module)
        and getattr(module, KV_CACHE_ATTR, None) is not None
    )


def _skip_static_kv_cache_validation(kv_cache_scheme: QuantizationArgs | None) -> bool:
    if kv_cache_scheme is None:
        return True

    return getattr(kv_cache_scheme, "dynamic", None) not in _STATIC_DYNAMICS


def _module_is_frozen(module: Module) -> bool:
    return getattr(module, "quantization_status", None) == QuantizationStatus.FROZEN


def _observer_num_observations(observer: Module | None) -> int:
    return int(getattr(observer, "num_observations", 0) or 0)


def _get_invalid_scale_reason(scale: torch.Tensor | None) -> str | None:
    if scale is None:
        return "missing"

    scale = (
        scale.detach() if isinstance(scale, torch.Tensor) else torch.as_tensor(scale)
    )
    if scale.is_meta:
        return "stored on meta device"
    if scale.numel() == 0:
        return "empty"

    scale = scale.to(device="cpu")
    if not torch.isfinite(scale).all().item():
        return "non-finite"
    if (scale <= 0).any().item():
        return "non-positive"

    return None
