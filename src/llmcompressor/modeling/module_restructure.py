from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from compressed_tensors.offload.module import remove_module_offload
from torch.nn import Module


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


@torch.no_grad()
def transfer_module_state(
    target: Module,
    source: Module,
    device: torch.device,
    *,
    recurse: bool = True,
) -> None:
    """Adopt source storage while preserving target tensor identities and new state."""
    # Dispatch also attaches caches to parameterless target containers.
    for module in target.modules() if recurse else (target,):
        remove_module_offload(module, onload_tensors=True)
    source_state = _named_state(source, recurse=recurse)
    target_state = _named_state(target, recurse=recurse)
    missing_source_names = set(source_state) - set(target_state)
    if missing_source_names:
        raise ValueError(
            "Restructured target is missing source state entries: "
            f"{sorted(missing_source_names)}"
        )
    for name, target_tensor in target_state.items():
        if not target_tensor.is_meta:
            # Shared state may already have been materialized and calibrated.
            # Preserve its values instead of replacing them with stale source data.
            if target_tensor.device != device:
                _adopt_tensor(target_tensor, target_tensor, device)
            continue
        source_tensor = source_state.get(name)
        if source_tensor is None:
            # New quantization parameters and observer buffers have no source.
            source_tensor = torch.full_like(
                target_tensor, 1 if name.endswith("_scale") else 0, device=device
            )
        elif target_tensor.shape != source_tensor.shape:
            raise ValueError(
                f"Cannot materialize {name}: target shape "
                f"{tuple(target_tensor.shape)} differs from source shape "
                f"{tuple(source_tensor.shape)}"
            )
        _adopt_tensor(target_tensor, source_tensor, device)

    for name, source_module in source.named_modules() if recurse else [("", source)]:
        target_module = target.get_submodule(name)
        for attr, value in vars(source_module).items():
            if attr not in {"_parameters", "_buffers", "_modules"}:
                _transfer_attribute(getattr(target_module, attr, None), value, device)

    if not recurse:
        # Modifier initialization can add observers with meta buffers. Existing
        # structural children are materialized through their own source mappings.
        for name, child in target.named_children():
            if source._modules.get(name) is None:
                transfer_module_state(child, Module(), device)


def _adopt_tensor(target, source, device):
    # detach/to and Parameter wrapping share storage when device and dtype match.
    value = source.detach().to(device=device, dtype=target.dtype)
    if value.is_inference() and not torch.is_inference_mode_enabled():
        # Keep the storage while allowing normal Parameter initialization.
        value = torch.empty(0, device=value.device, dtype=value.dtype).set_(value)
    if isinstance(target, torch.nn.Parameter):
        value = torch.nn.Parameter(value, requires_grad=target.requires_grad)
    else:
        value.requires_grad_(target.requires_grad)
    value.__dict__.update(target.__dict__)
    torch.utils.swap_tensors(target, value)


def _transfer_attribute(target, source, device):
    """Adopt tensor-valued attributes while preserving aliases to registered state."""
    if isinstance(target, torch.Tensor) and target.is_meta:
        _adopt_tensor(target, source, device)
    elif isinstance(target, dict) and isinstance(source, dict):
        for key in target.keys() & source.keys():
            _transfer_attribute(target[key], source[key], device)
    elif isinstance(target, (list, tuple)) and isinstance(source, (list, tuple)):
        for left, right in zip(target, source):
            _transfer_attribute(left, right, device)


def _named_state(module: Module, *, recurse: bool = True) -> dict[str, torch.Tensor]:
    state = dict(module.named_parameters(recurse=recurse, remove_duplicate=False))
    state.update(dict(module.named_buffers(recurse=recurse, remove_duplicate=False)))
    return state
