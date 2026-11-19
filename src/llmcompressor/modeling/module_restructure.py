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


RESTRUCTURE_METADATA_ATTR = "_llmcompressor_restructure"


def copy_module_state(
    target: Module,
    source: Module,
    device: torch.device,
) -> None:
    source_state = _named_state(source)
    target_meta_state = _named_state(target)
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
        target_state = _named_state(module)
        for name, target_tensor in target_state.items():
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

    materialize_meta_module(target, device=device, initializer=initialize)


def _named_state(module: Module) -> dict[str, torch.Tensor]:
    state = dict(module.named_parameters(remove_duplicate=False))
    state.update(dict(module.named_buffers(remove_duplicate=False)))
    return state
