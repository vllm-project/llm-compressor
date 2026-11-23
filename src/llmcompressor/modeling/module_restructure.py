from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from torch.nn import Module

from llmcompressor.utils.pytorch.meta import (
    materialize_meta_module,
)


class ModuleRestructure(ABC):
    """Select modules and restructure either their meta or real copies."""

    @abstractmethod
    def targets(self, model: Module) -> Iterable[tuple[str, Module, Any]]:
        """Find modules and the information needed to restructure them."""

    @abstractmethod
    def restructure(self, source: Module, context: Any) -> Module:
        """Restructure a module using the same operation for meta and real tensors."""


def copy_module_state(
    target: Module,
    source: Module,
    device: torch.device,
    *,
    recurse: bool = True,
) -> None:
    source_state = _named_state(source, recurse=recurse)
    target_meta_state = _named_state(target, recurse=recurse)
    missing_source_names = set(source_state) - set(target_meta_state)
    if missing_source_names:
        raise ValueError(
            "Restructured target is missing source state entries: "
            f"{sorted(missing_source_names)}"
        )
    meta_names = {
        name
        for name, tensor in target_meta_state.items()
        if tensor.device.type == "meta"
    }

    def initialize(module: Module) -> None:
        target_state = _named_state(module, recurse=recurse)
        for name, target_tensor in target_state.items():
            # Shared parameters may already have been materialized and calibrated
            # through another module. Never overwrite them with stale source data.
            if name not in meta_names:
                continue
            source_tensor = source_state.get(name)
            if source_tensor is None:
                if name in meta_names:
                    target_tensor.fill_(1 if name.endswith("_scale") else 0)
                continue
            if source_tensor.device.type == "meta":
                raise ValueError(f"Source state {name} is still on the meta device")
            if target_tensor.shape != source_tensor.shape:
                raise ValueError(
                    f"Cannot materialize {name}: target shape "
                    f"{tuple(target_tensor.shape)} differs from source shape "
                    f"{tuple(source_tensor.shape)}"
                )
            target_tensor.copy_(
                source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            )
        for name, source_module in (
            source.named_modules() if recurse else [("", source)]
        ):
            target_module = module.get_submodule(name)
            for attr, value in vars(source_module).items():
                if attr not in {"_parameters", "_buffers", "_modules"}:
                    _materialize_attribute(
                        getattr(target_module, attr, None), value, device
                    )

    materialize_meta_module(
        target, device=device, initializer=initialize, recurse=recurse
    )


def _materialize_attribute(target, source, device):
    """Fill tensor-valued attributes while preserving aliases to registered state."""
    if isinstance(target, torch.Tensor) and target.device.type == "meta":
        if not isinstance(source, torch.Tensor) or source.device.type == "meta":
            raise ValueError("Cannot materialize tensor attribute without real state")
        value = source.detach().to(device=device, dtype=target.dtype).clone()
        if isinstance(target, torch.nn.Parameter):
            value = torch.nn.Parameter(value, requires_grad=target.requires_grad)
        torch.utils.swap_tensors(target, value)
    elif isinstance(target, dict) and isinstance(source, dict):
        for key in target.keys() & source.keys():
            _materialize_attribute(target[key], source[key], device)
    elif isinstance(target, (list, tuple)) and isinstance(source, (list, tuple)):
        for left, right in zip(target, source):
            _materialize_attribute(left, right, device)


def _named_state(module: Module, *, recurse: bool = True) -> dict[str, torch.Tensor]:
    state = dict(module.named_parameters(recurse=recurse, remove_duplicate=False))
    state.update(dict(module.named_buffers(recurse=recurse, remove_duplicate=False)))
    return state
