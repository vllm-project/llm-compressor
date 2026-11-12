"""Meta-model preparation and source-to-target materialization for sequential runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from torch.nn import Module

from llmcompressor.utils.pytorch.meta import (
    copy_module_to_meta,
    materialize_meta_module,
)

__all__ = [
    "MaterializationPlan",
    "MetaModel",
    "MetaTransform",
]


Materializer = Callable[
    [Module, Mapping[str, Module], Mapping[str, dict], torch.device],
    dict[str, dict],
]


@dataclass
class MaterializationEntry:
    """How to populate one target module from one or more source modules."""

    target_name: str
    source_names: tuple[str, ...]
    materialize: Materializer
    include_descendants: bool


class MetaTransform(Protocol):
    """A graph transform with a meta phase and a per-subgraph materialization phase."""

    def apply_to_meta(
        self, model: Module, plan: MaterializationPlan
    ) -> None: ...


class MaterializationPlan:
    """Map transformed meta modules to the real modules that provide their weights."""

    def __init__(self, source_model: Module, meta_model: Module):
        self.source_model = source_model
        self.meta_model = meta_model
        self.entries: dict[str, MaterializationEntry] = {}

        # Identity mappings are the base case. A structural transform can replace
        # these entries with mappings from its target modules to their source tensors.
        self.register("", ("",), _copy_module_state)
        for name, _ in meta_model.named_modules():
            if not name or not _has_submodule(source_model, name):
                continue
            self.register(name, (name,), _copy_module_state)

    def register(
        self,
        target_name: str,
        source_names: Sequence[str],
        materialize: Materializer,
        *,
        include_descendants: bool = False,
    ) -> None:
        """Register or replace the source mapping for a target module subtree."""
        self.entries[target_name] = MaterializationEntry(
            target_name=target_name,
            source_names=tuple(source_names),
            materialize=materialize,
            include_descendants=include_descendants,
        )

    def remove_subtree(self, target_name: str) -> None:
        """Remove identity mappings below a target path before a graph transform."""
        prefix = f"{target_name}." if target_name else ""
        for name in list(self.entries):
            if name == target_name or name.startswith(prefix):
                del self.entries[name]

    def _entries_for(self, target_modules: Mapping[str, Module]):
        structural = [
            entry
            for name, entry in self.entries.items()
            if entry.include_descendants
            and (
                name in target_modules
                or any(
                    not name or target_name.startswith(f"{name}.")
                    for target_name in target_modules
                )
            )
        ]
        selected_structural: list[MaterializationEntry] = []
        for entry in sorted(structural, key=lambda item: item.target_name.count(".")):
            if any(
                _is_same_or_child(entry.target_name, parent.target_name)
                for parent in selected_structural
            ):
                continue
            selected_structural.append(entry)

        direct = [
            entry
            for name, entry in self.entries.items()
            if not entry.include_descendants and name in target_modules
            and not any(
                _is_same_or_child(name, parent.target_name)
                for parent in selected_structural
            )
        ]
        return selected_structural + direct

    def source_modules_for(
        self, target_modules: Mapping[str, Module]
    ) -> dict[str, Module]:
        """Return the source modules needed to materialize a target subgraph."""
        modules: dict[str, Module] = {}
        for entry in self._entries_for(target_modules):
            for source_name in entry.source_names:
                source = _get_submodule(self.source_model, source_name)
                if entry.include_descendants:
                    for relative_name, submodule in source.named_modules():
                        qualified_name = (
                            f"{source_name}.{relative_name}"
                            if source_name and relative_name
                            else source_name or relative_name
                        )
                        modules[qualified_name] = submodule
                else:
                    modules[source_name] = source
        return modules

    def materialize_subgraph(
        self,
        target_modules: Mapping[str, Module],
        source_modules: Mapping[str, Module],
        source_offload_kwargs: Mapping[str, dict],
        device: torch.device,
    ) -> dict[str, dict]:
        """Populate target modules in place and attach their offload settings."""
        target_offload_kwargs: dict[str, dict] = {}
        for entry in self._entries_for(target_modules):
            target = target_modules.get(
                entry.target_name, _get_submodule(self.meta_model, entry.target_name)
            )
            sources = {
                name: source_modules[name]
                for name in entry.source_names
                if name in source_modules
            }
            target_kwargs = entry.materialize(
                target, sources, source_offload_kwargs, device
            )
            for relative_name, kwargs in target_kwargs.items():
                target_name = (
                    f"{entry.target_name}.{relative_name}"
                    if entry.target_name and relative_name
                    else entry.target_name or relative_name
                )
                target_offload_kwargs[target_name] = kwargs
        return target_offload_kwargs


class MetaModel:
    """A meta-initialized calibration graph backed by a separate source model."""

    def __init__(
        self,
        source_model: Module,
        transforms: Sequence[MetaTransform] = (),
    ):
        self.source_model = source_model
        self.model = copy_module_to_meta(source_model)
        self.materialization = MaterializationPlan(source_model, self.model)
        for transform in transforms:
            self.apply_transform(transform)

    def apply_transform(self, transform: MetaTransform) -> None:
        """Apply a graph transform and record how its tensors will be populated."""
        transform.apply_to_meta(self.model, self.materialization)

    def source_modules_for(
        self, target_modules: Mapping[str, Module]
    ) -> dict[str, Module]:
        return self.materialization.source_modules_for(target_modules)

    def materialize_subgraph(
        self,
        target_modules: Mapping[str, Module],
        source_modules: Mapping[str, Module],
        source_offload_kwargs: Mapping[str, dict],
        device: torch.device,
    ) -> dict[str, dict]:
        return self.materialization.materialize_subgraph(
            target_modules, source_modules, source_offload_kwargs, device
        )


def _copy_module_state(
    target: Module,
    sources: Mapping[str, Module],
    source_offload_kwargs: Mapping[str, dict],
    device: torch.device,
) -> dict[str, dict]:
    """Identity transform: copy same-named parameters and buffers into the target."""
    if len(sources) != 1:
        raise ValueError("identity materialization requires exactly one source module")
    source_name, source = next(iter(sources.items()))
    meta_tensor_names = {
        name for name, tensor in _local_tensors(target).items()
        if tensor.device.type == "meta"
    }

    def initialize(materialized: Module) -> None:
        source_state = _local_tensors(source)
        target_state = _local_tensors(materialized)
        for name, target_tensor in target_state.items():
            source_tensor = source_state.get(name)
            if source_tensor is None:
                # Modifier-created observers and qparams have no source checkpoint
                # tensor. Give them deterministic initial values before calibration.
                if name in meta_tensor_names:
                    if name.endswith("_scale"):
                        target_tensor.fill_(1)
                    else:
                        target_tensor.zero_()
                continue
            if target_tensor.shape != source_tensor.shape:
                raise ValueError(
                    f"Cannot materialize {name}: target shape {tuple(target_tensor.shape)} "
                    f"does not match source shape {tuple(source_tensor.shape)}"
                )
            target_tensor.copy_(
                source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            )

    if meta_tensor_names:
        materialize_meta_module(
            target, device=device, initializer=initialize, recurse=False
        )

    kwargs = source_offload_kwargs.get(
        source_name, get_cache_init_kwargs(source)
    )
    offload_module(target, **kwargs)
    return {"": kwargs}


def _local_tensors(module: Module) -> dict[str, torch.Tensor]:
    tensors = {
        name: tensor for name, tensor in module._parameters.items() if tensor is not None
    }
    tensors.update(
        {name: tensor for name, tensor in module._buffers.items() if tensor is not None}
    )
    return tensors


def _has_submodule(module: Module, name: str) -> bool:
    try:
        _get_submodule(module, name)
    except (AttributeError, KeyError):
        return False
    return True


def _get_submodule(module: Module, name: str) -> Module:
    return module.get_submodule(name) if name else module


def _is_same_or_child(name: str, parent: str) -> bool:
    return not parent or name == parent or name.startswith(f"{parent}.")
