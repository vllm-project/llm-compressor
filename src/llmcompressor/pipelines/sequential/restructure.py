"""Prototype whole-model meta initialization and delayed restructuring."""

from __future__ import annotations

import copy
from types import MethodType
from typing import Any

import torch
from compressed_tensors.offload import offload_module, set_onload_device
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import (
    remove_module_offload,
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from torch.nn import Module, Parameter

from llmcompressor.modeling.decompression import DecompressionRestructure
from llmcompressor.modeling.module_restructure import (
    ModuleRestructure,
    copy_module_state,
)
from llmcompressor.modeling.moe.linearize import MoELinearizationRestructure
from llmcompressor.modeling.offset_norm import (
    NormCalibrationModule,
    get_norm_calibration_targets,
)
from llmcompressor.utils.dev import get_main_device

__all__ = ["RestructureHandler", "restructure_requires_restore"]


def restructure_requires_restore(target: Module) -> bool:
    return isinstance(target, NormCalibrationModule)


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


class RestructureHandler:
    """Retain real sources while the live model initializes and traces on meta.

    ``mapping`` maps current model modules to their temporary source holders.
    Ordinary entries describe direct state; restructure entries own a whole subtree.
    """

    def __init__(self, model: Module):
        self.model = model
        self.mapping: dict[Module, Module] = {}
        self.restructures: dict[Module, tuple[ModuleRestructure, Any]] = {}
        self.restore_sources: dict[Module, tuple[Module, dict]] = {}
        self.source_root: Module | None = None

    def delay_loading(self, module: Module, memo: dict[int, torch.Tensor]) -> Module:
        """Move direct tensor ownership to a copied holder, leaving meta tensors."""
        source = copy.copy(module)
        for name, value in vars(module).items():
            if isinstance(value, MethodType) and value.__self__ is module:
                source.__dict__[name] = value.__func__.__get__(source)
            elif (
                "hook" in name and isinstance(value, dict)
            ) or name == "_non_persistent_buffers_set":
                source.__dict__[name] = copy.copy(value)
        # Cached storage is independent of its module, but the dispatch wrapper
        # captures the original. Rebind the underlying forward to the holder.
        # Source onload removes the cache before conversion uses this forward.
        if isinstance(source._parameters, OffloadCache):
            if hasattr(source, "_original_forward_func"):
                source.forward = source._original_forward_func.__get__(source)
            remove_module_offload(module, onload_tensors=False)

        def to_meta(tensor, parameter=False):
            if id(tensor) not in memo:
                empty = torch.empty_like(tensor, device="meta")
                memo[id(tensor)] = (
                    Parameter(empty, requires_grad=tensor.requires_grad)
                    if parameter or isinstance(tensor, Parameter)
                    else empty
                )
            return memo[id(tensor)]

        for name in ("_parameters", "_buffers"):
            state = getattr(source, name)
            state = getattr(state, "offloaded_values", state)
            setattr(
                module,
                name,
                {
                    key: to_meta(tensor, name == "_parameters")
                    if tensor is not None
                    else None
                    for key, tensor in state.items()
                },
            )
        for name, value in list(vars(module).items()):
            if name not in {"_parameters", "_buffers", "_modules"}:
                module.__dict__[name] = _map_tensors(value, to_meta)
        self.mapping[module] = source
        return module

    def convert_model_to_meta(self) -> None:
        """Retain a real source tree and replace the live model's tensors with meta."""
        nodes = list(self.model.modules())
        memo: dict[int, torch.Tensor] = {}
        for module in nodes:
            self.delay_loading(module, memo)
        for module in nodes:
            self.mapping[module]._modules = {
                name: self.mapping[child] if child is not None else None
                for name, child in module._modules.items()
            }
        self.source_root = self.mapping[self.model]

    def _replace_module(self, old: Module, new: Module) -> None:
        for parent in list(self.model.modules()):
            for name, child in parent._modules.items():
                if child is old:
                    parent._modules[name] = new

    def apply_restructuring(self) -> None:
        """Restructure the meta model and associate replacements with real sources."""
        roots: set[str] = set()
        for transform in (
            MoELinearizationRestructure(),
            OffsetNormRestructure(),
            DecompressionRestructure(),
        ):
            for fqn, module, context in transform.targets(self.model):
                if any(
                    fqn == root
                    or fqn.startswith(f"{root}.")
                    or root.startswith(f"{fqn}.")
                    for root in roots
                ):
                    continue
                source = self.mapping[module]
                old_modules = list(module.modules())
                target = transform.restructure(module, context)
                for old in old_modules:
                    self.mapping.pop(old, None)
                self.mapping[target] = source
                self.restructures[target] = (transform, context)
                self._replace_module(module, target)
                roots.add(fqn)

    def configure_onload(self, device: torch.device) -> None:
        if self.source_root is not None:
            set_onload_device(self.source_root, device)

    @staticmethod
    def _release_source(source: Module, recurse: bool) -> None:
        for module in source.modules() if recurse else (source,):
            module._parameters = {}
            module._buffers = {}
            for name, value in list(vars(module).items()):
                if name not in {"_parameters", "_buffers", "_modules"}:
                    module.__dict__[name] = _map_tensors(value, lambda tensor: None)

    def _materialize_added_children(self, target: Module, device: torch.device) -> None:
        # Modifier initialization can add observers with meta buffers. They have
        # no retained source, and must initialize alongside their owning module.
        for child in target.children():
            if child not in self.mapping:
                copy_module_state(child, Module(), device, recurse=False)
                self._materialize_added_children(child, device)

    def onload_and_restructure_modules(
        self, modules: dict[str, Module], device: torch.device
    ) -> dict[str, dict]:
        pending = {
            name: target for name, target in modules.items() if target in self.mapping
        }
        sources = {}
        for name, target in pending.items():
            source = self.mapping[target]
            if target in self.restructures:
                sources.update(source.named_modules(prefix=name))
            else:
                sources[name] = source
        kwargs = subgraph_onload_modules(sources)
        for name, target in pending.items():
            source = self.mapping.pop(target)
            record = self.restructures.pop(target, None)
            remove_module_offload(target, onload_tensors=True)
            if record is None:
                copy_module_state(target, source, device, recurse=False)
                self._materialize_added_children(target, device)
            else:
                transform, context = record
                converted = transform.restructure(source, context)
                copy_module_state(target, converted, device)
                if restructure_requires_restore(target):
                    self.restore_sources[target] = (source, kwargs[name])
            if target not in self.restore_sources:
                self._release_source(source, recurse=record is not None)
        return kwargs

    def materialize_all(self) -> None:
        device = get_main_device()
        self.configure_onload(device)
        modules = {
            name: module
            for name, module in self.model.named_modules()
            if module in self.mapping
        }
        kwargs = subgraph_onload_modules(modules)
        kwargs.update(self.onload_and_restructure_modules(modules, device))
        subgraph_offload_modules(modules, kwargs)

    def restore_restructured_modules(
        self, modules: dict[str, Module] | None = None, *, offload_sources=True
    ) -> None:
        for target, (source, kwargs) in list(self.restore_sources.items()):
            if modules is not None and target not in modules.values():
                continue
            remove_module_offload(target, onload_tensors=True)
            remove_module_offload(source, onload_tensors=True)
            target.restore(source)
            if offload_sources:
                offload_module(source, **kwargs)
            self._replace_module(target, source)
            if modules is not None:
                for name, module in modules.items():
                    if module is target:
                        modules[name] = source
            del self.restore_sources[target]

    def finish(self) -> None:
        """Restore a complete usable model, including state outside traced graphs."""
        self.materialize_all()
        self.restore_restructured_modules()
        self.mapping.clear()
        self.restructures.clear()
        self.source_root = None


class OffsetNormRestructure(ModuleRestructure):
    def restructure(self, source: Module, context) -> Module:
        class_name, config = context
        return NormCalibrationModule.load_from_registry(
            class_name, original=source, config=config
        )

    def targets(self, model: Module):
        targets = get_norm_calibration_targets(model)
        if not targets:
            return
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("Offset norm restructures require a model config")
        for fqn, source, class_name in targets:
            yield fqn, source, (class_name, config)
