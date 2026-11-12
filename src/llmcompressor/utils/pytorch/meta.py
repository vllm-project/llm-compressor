"""Helpers for copying and materializing individual modules on the meta device."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, TypeVar

import torch
from compressed_tensors.offload.module import remove_module_offload
from torch.nn import Module, Parameter

__all__ = ["copy_module_to_meta", "materialize_meta_module"]

ModuleT = TypeVar("ModuleT", bound=Module)


def copy_module_to_meta(module: ModuleT) -> ModuleT:
    """Deep-copy one module tree with all tensor state replaced by meta tensors."""
    memo: dict[int, Any] = {}
    visited: set[int] = set()

    def memoize_tensors(value: Any) -> None:
        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)

        if isinstance(value, torch.Tensor):
            empty = torch.empty_like(value, device="meta")
            memo[value_id] = (
                Parameter(empty, requires_grad=value.requires_grad)
                if isinstance(value, Parameter)
                else empty
            )
        elif isinstance(value, dict):
            for key, item in value.items():
                memoize_tensors(key)
                memoize_tensors(item)
        elif isinstance(value, (tuple, list, set)):
            for item in value:
                memoize_tensors(item)

    for submodule in module.modules():
        for mapping in (submodule._parameters, submodule._buffers):
            values = getattr(mapping, "offloaded_values", None)
            for tensor in (values if values is not None else mapping).values():
                if tensor is not None:
                    memoize_tensors(tensor)

        for name, value in submodule.__dict__.items():
            if name not in {"_parameters", "_buffers", "_modules"}:
                memoize_tensors(value)

    meta_module = copy.deepcopy(module, memo)
    for submodule in meta_module.modules():
        remove_module_offload(submodule, onload_tensors=False)

    # Offload caches can store CPU tensors. The direct cache storage was memoized
    # above, so this pass should only be needed for unusual cache implementations.
    for submodule in meta_module.modules():
        for mapping_name in ("_parameters", "_buffers"):
            mapping = getattr(submodule, mapping_name)
            for name, tensor in mapping.items():
                if tensor is not None and tensor.device.type != "meta":
                    empty = torch.empty_like(tensor, device="meta")
                    mapping[name] = (
                        Parameter(empty, requires_grad=tensor.requires_grad)
                        if isinstance(tensor, Parameter)
                        else empty
                    )

    return meta_module


@torch.no_grad()
def materialize_meta_module(
    module: ModuleT,
    *,
    device: torch.device | str,
    initializer: Callable[[ModuleT], None],
    recurse: bool = True,
) -> None:
    """Allocate storage for meta parameters and buffers, then fill them in place."""
    originals = []
    has_meta_tensors = False
    submodules = module.modules() if recurse else (module,)
    for submodule in submodules:
        for mapping_name in ("_parameters", "_buffers"):
            mapping = getattr(submodule, mapping_name)
            for name, tensor in mapping.items():
                if tensor is None:
                    continue
                if tensor.device.type == "meta":
                    has_meta_tensors = True
                    saved_value = None
                else:
                    saved_value = tensor.detach().clone()
                originals.append(
                    (submodule, mapping_name, name, tensor, saved_value)
                )

    if not has_meta_tensors:
        raise ValueError("module has no meta parameters or buffers to materialize")

    swap_params = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(True)
    try:
        module.to_empty(device=device, recurse=recurse)
    finally:
        torch.__future__.set_swap_module_params_on_conversion(swap_params)

    for submodule, mapping_name, name, original, saved_value in originals:
        mapping = getattr(submodule, mapping_name)
        materialized = mapping[name]
        if materialized is not original:
            torch.utils.swap_tensors(original, materialized)
            mapping[name] = original
            materialized = original
        if saved_value is not None:
            materialized.copy_(
                saved_value.to(device=materialized.device, dtype=materialized.dtype)
            )

    initializer(module)

    # ``to_empty`` only handles registered parameters and buffers. Tensor-valued
    # attributes are converted to meta by ``copy_module_to_meta`` too, so fail here
    # with a useful error instead of letting a later forward encounter meta state.
    for submodule in (module.modules() if recurse else (module,)):
        for name, value in submodule.__dict__.items():
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            if _contains_meta_tensor(value):
                raise ValueError(
                    "Cannot materialize unregistered tensor attribute "
                    f"{submodule.__class__.__name__}.{name}; register it as a "
                    "parameter or buffer, or materialize it in the transform"
                )


def _contains_meta_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device.type == "meta"
    if isinstance(value, dict):
        return any(
            _contains_meta_tensor(key) or _contains_meta_tensor(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set)):
        return any(_contains_meta_tensor(item) for item in value)
    return False
