from collections.abc import Iterable

import torch
from compressed_tensors.modeling import KV_CACHE_ATTR
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStatus,
    is_attention_module,
)
from compressed_tensors.utils import match_named_modules
from torch.nn import Module

__all__ = ["validate_static_kv_cache_scales"]

_STATIC_DYNAMICS = (False, None)
_KV_CACHE_BASE_NAMES = ("k", "v")


def validate_static_kv_cache_scales(
    model: Module,
    targets: set[str],
    ignore: list[str],
    kv_cache_scheme: QuantizationArgs | None,
) -> None:
    if _skip_static_kv_cache_validation(kv_cache_scheme):
        return

    failures = [
        failure
        for module_name, module in _iter_static_kv_cache_modules(model, targets, ignore)
        if not _is_frozen(module)
        for base_name in _KV_CACHE_BASE_NAMES
        if (failure := _kv_cache_scale_failure(module_name, module, base_name))
    ]
    if failures:
        raise ValueError(_format_static_kv_cache_error(failures))


def _iter_static_kv_cache_modules(
    model: Module,
    targets: set[str],
    ignore: list[str],
) -> Iterable[tuple[str, Module]]:
    for module_name, module in match_named_modules(model, targets, ignore):
        if _is_kv_cache_module(module):
            yield module_name, module


def _is_kv_cache_module(module: Module) -> bool:
    return (
        getattr(module, "quantization_scheme", None) is not None
        and is_attention_module(module)
        and getattr(module, KV_CACHE_ATTR, None) is not None
    )


def _skip_static_kv_cache_validation(kv_cache_scheme: QuantizationArgs | None) -> bool:
    if kv_cache_scheme is None:
        return True

    return getattr(kv_cache_scheme, "dynamic", None) not in _STATIC_DYNAMICS


def _is_frozen(module: Module) -> bool:
    return getattr(module, "quantization_status", None) == QuantizationStatus.FROZEN


def _kv_cache_scale_failure(
    module_name: str,
    module: Module,
    base_name: str,
) -> str | None:
    observer_name = f"{base_name}_observer"
    scale_name = f"{base_name}_scale"
    observer = getattr(module, observer_name, None)
    if observer is None or not getattr(observer, "has_statistics", False):
        return f"{_format_module_attr(module_name, observer_name)} (missing/unobserved)"

    invalid_reason = _get_invalid_scale_reason(getattr(module, scale_name, None))
    if invalid_reason is None:
        return None

    return f"{_format_module_attr(module_name, scale_name)} ({invalid_reason})"


def _format_static_kv_cache_error(failures: list[str]) -> str:
    return (
        "KV-cache quantization calibration failed. Static KV-cache scales must "
        "be finite positive values before saving a quantized checkpoint. Found "
        f"{', '.join(failures)}. This usually means K/V tensors were not "
        "observed during calibration. That can happen when a model bypasses the "
        "standard cache update(...) API; in that case, update the model upstream "
        "to route KV-cache writes through the standard cache update path."
    )


def _format_module_attr(module_name: str, attr_name: str) -> str:
    return f"{module_name}.{attr_name}" if module_name else attr_name


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
