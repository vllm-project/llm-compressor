"""Helpers for meta-copying and materializing PyTorch modules."""

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
    """Copy a module tree while replacing tensor storage with meta tensors.

    Tensor identities are memoized before ``deepcopy`` so tied parameters and
    buffers remain tied in the copy without duplicating their storage.
    """
    memo: dict[int, Any] = {}
    visited: set[int] = set()

    def memoize_tensor_tree(value: Any) -> None:
        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)

        if isinstance(value, torch.Tensor):
            if isinstance(value, Parameter):
                meta_value = Parameter(
                    torch.empty_like(value, device="meta"),
                    requires_grad=value.requires_grad,
                )
            else:
                meta_value = torch.empty_like(value, device="meta")
            memo[value_id] = meta_value
        elif isinstance(value, dict):
            for key, item in value.items():
                memoize_tensor_tree(key)
                memoize_tensor_tree(item)
        elif isinstance(value, (tuple, list, set)):
            for item in value:
                memoize_tensor_tree(item)

    for submodule in module.modules():
        for tensor in (*submodule._parameters.values(), *submodule._buffers.values()):
            if tensor is not None:
                memoize_tensor_tree(tensor)

        for name, value in submodule.__dict__.items():
            if name not in {"_parameters", "_buffers", "_modules"}:
                memoize_tensor_tree(value)

    meta_module = copy.deepcopy(module, memo)
    # Offload hooks copied from the source model own its cache and must not be
    # triggered by calibration forwards on the structure-only model.
    for submodule in meta_module.modules():
        remove_module_offload(submodule, onload_tensors=False)
    return meta_module


@torch.no_grad()
def materialize_meta_module(
    module: ModuleT,
    *,
    device: torch.device | str,
    initializer: Callable[[ModuleT], None],
    recurse: bool = True,
) -> None:
    """Allocate module storage and fill it through ``initializer``.

    ``to_empty`` replaces storage for all parameters and buffers, including tensors
    that already have values. Non-meta state is saved and restored. Parameter
    identities are kept stable for hooks and other objects holding direct references.
    """
    preserved_tensors = []
    has_meta_tensors = False
    submodules = module.modules() if recurse else (module,)
    for submodule in submodules:
        for mapping_name in ("_parameters", "_buffers"):
            tensor_mapping = getattr(submodule, mapping_name)
            for name, tensor in tensor_mapping.items():
                if tensor is None:
                    continue
                if tensor.device.type == "meta":
                    has_meta_tensors = True
                else:
                    preserved_tensors.append(
                        (
                            submodule,
                            mapping_name,
                            name,
                            tensor,
                            tensor.detach().clone(),
                        )
                    )

    if not has_meta_tensors:
        raise ValueError("module has no meta parameters or buffers to materialize")

    swap_params = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(True)
    try:
        module.to_empty(device=device, recurse=recurse)
    finally:
        torch.__future__.set_swap_module_params_on_conversion(swap_params)

    for submodule, mapping_name, name, original, saved_value in preserved_tensors:
        tensor_mapping = getattr(submodule, mapping_name)
        materialized = tensor_mapping[name]
        if materialized is not original:
            torch.utils.swap_tensors(original, materialized)
            tensor_mapping[name] = original
            materialized = original
        materialized.copy_(
            saved_value.to(device=materialized.device, dtype=materialized.dtype)
        )

    initializer(module)
