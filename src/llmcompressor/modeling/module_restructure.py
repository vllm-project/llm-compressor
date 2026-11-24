from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from compressed_tensors.offload.module import remove_module_offload
from torch.nn import Module

from llmcompressor.utils.pytorch.meta import (
    materialize_meta_module,
)


class ModuleRestructure(ABC):
    """Track the real sources affected by one model-wide restructuring operation."""

    def __init__(self):
        self.mapping: dict[Module, Module] = {}  # real source -> live target
        self.delayed: dict[Module, Module] = {}  # handler's live target -> source

    @abstractmethod
    def apply(
        self,
        model: Module,
        mapping: dict[Module, Module] | None = None,
    ) -> Module:
        """Restructure the model and remember sources when loading is delayed."""

    @abstractmethod
    def restructure_sources(
        self,
        model: Module,
        sources: dict[str, Module],
        device: torch.device,
        offload_kwargs: dict[str, dict],
    ) -> None:
        """Restructure onloaded sources, transferring replacement state to targets."""

    def restore(
        self,
        model: Module,
        modules: dict[str, Module] | None = None,
    ) -> Module:
        """Restore temporary conversions, leaving permanent restructures unchanged."""
        return model


def _map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        mapped = value.copy()
        for key, item in value.items():
            mapped[key] = _map_tensors(item, function)
        return mapped
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    if isinstance(value, set):
        return {_map_tensors(item, function) for item in value}
    return value


def clear_module_state(module: Module) -> None:
    """Release a consumed source's direct tensors, leaving its children alone."""
    module._parameters = {}
    module._buffers = {}
    for name, value in list(vars(module).items()):
        if name not in {"_parameters", "_buffers", "_modules"}:
            module.__dict__[name] = _map_tensors(value, lambda tensor: None)


def replace_module(model: Module, source: Module, target: Module) -> Module:
    """Replace a module, including every shared alias and the root itself."""
    if model is source:
        return target
    if source is not target:
        for parent in list(model.modules()):
            for name, child in parent._modules.items():
                if child is source:
                    parent._modules[name] = target
    return model


def copy_module_state(
    target: Module,
    source: Module,
    device: torch.device,
    *,
    recurse: bool = True,
) -> None:
    remove_module_offload(target, onload_tensors=True)
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
    if not recurse:
        # Modifier initialization can add observers with meta buffers. Existing
        # structural children are materialized through their own source mappings.
        for name, child in target.named_children():
            if source._modules.get(name) is None:
                copy_module_state(child, Module(), device)


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
