from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy

import torch
from torch.nn import Module


class ModuleRestructure(ABC):
    """Apply one model-wide restructure and repeat it on selected real sources."""

    requires_restore: bool = False

    @abstractmethod
    def apply(
        self,
        model: Module,
        mapping: dict[Module, dict] | None = None,
    ) -> Module:
        """Restructure the model and remember sources when loading is delayed."""

    @abstractmethod
    def restructure_sources(
        self,
        model: Module,
        targets: dict[Module, dict],
        device: torch.device,
        offload_kwargs: dict[str, dict],
    ) -> None:
        """Restructure onloaded sources, transferring replacement state to targets."""

    def restore(
        self,
        model: Module,
        targets: dict[Module, dict],
        modules: dict[str, Module] | None = None,
    ) -> Module:
        """Restore temporary conversions, leaving permanent restructures unchanged."""
        return model


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
) -> None:
    """Adopt direct source storage while preserving target tensors and their aliases.

    Sources must already have the target device/dtype.
    """
    for state_name in ("_parameters", "_buffers"):
        source_state = getattr(source, state_name)
        target_state = getattr(target, state_name)
        for name, placeholder in target_state.items():
            value = source_state.pop(name, None)
            if placeholder is None or not placeholder.is_meta:
                continue
            if value is None:
                # Recipe initialization adds qparams only to the target.
                value = torch.nn.Parameter(
                    torch.full_like(
                        placeholder, 1 if name.endswith("_scale") else 0, device=device
                    ),
                    requires_grad=placeholder.requires_grad,
                )
            elif placeholder.shape != value.shape:
                raise ValueError(
                    f"Cannot materialize {name}: target shape "
                    f"{tuple(placeholder.shape)} differs from source shape "
                    f"{tuple(value.shape)}"
                )
            else:
                # Other source holders may still reference this tensor. Share its
                # storage without swapping the original source object to meta.
                value = copy(value)
                value.requires_grad_(placeholder.requires_grad)
            value.__dict__ = placeholder.__dict__.copy()
            torch.utils.swap_tensors(placeholder, value)
        source_state.clear()
