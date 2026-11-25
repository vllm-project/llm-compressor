from __future__ import annotations

from abc import ABC, abstractmethod

import torch
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
    aliases: dict[torch.Tensor, list[tuple[dict, str]]] | None = None,
) -> None:
    """Move direct state; sources must already have the target device/dtype."""
    for state_name in ("_parameters", "_buffers"):
        source_state = getattr(source, state_name)
        target_state = getattr(target, state_name)
        for name, placeholder in target_state.items():
            value = source_state.pop(name, None)
            if placeholder is None or not placeholder.is_meta:
                continue
            if value is None:
                # Recipe initialization adds qparams only to the target.
                value = torch.full_like(
                    placeholder,
                    1 if name.endswith("_scale") else 0,
                    device=device,
                    requires_grad=placeholder.requires_grad,
                )
                if state_name == "_parameters":
                    value = torch.nn.Parameter(
                        value, requires_grad=placeholder.requires_grad
                    )
            elif placeholder.shape != value.shape:
                raise ValueError(
                    f"Cannot materialize {name}: target shape "
                    f"{tuple(placeholder.shape)} differs from source shape "
                    f"{tuple(value.shape)}"
                )
            else:
                value.requires_grad_(placeholder.requires_grad)
            target_state[name] = value
            if aliases is not None:
                for state, alias_name in aliases.pop(placeholder, ()):
                    if state.get(alias_name) is placeholder:
                        state[alias_name] = value
        source_state.clear()
