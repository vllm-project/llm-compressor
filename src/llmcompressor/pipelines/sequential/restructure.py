"""Prototype whole-model meta initialization and delayed restructuring."""

from __future__ import annotations

import copy

import torch
from compressed_tensors.offload import (
    get_cache_init_kwargs,
    offload_module,
    set_onload_device,
)
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import (
    remove_module_offload,
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from torch.nn import Module, Parameter

from llmcompressor.modeling.decompression import DecompressionRestructure
from llmcompressor.modeling.module_restructure import (
    copy_module_state,
    replace_module,
)
from llmcompressor.modeling.moe.linearize import MoELinearizationRestructure
from llmcompressor.modeling.offset_norm import (
    NormCalibrationModule,
    OffsetNormRestructure,
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
        config = getattr(model, "config", None)
        self.restructures = (
            DecompressionRestructure(),
            MoELinearizationRestructure(config),
            OffsetNormRestructure(config),
        )
        self.restructured_modules: set[Module] = set()
        self.restore_sources: dict[Module, tuple[Module, dict]] = {}
        self.source_root: Module | None = None

    def delay_loading(self, module: Module, memo: dict[int, torch.Tensor]) -> Module:
        """Move direct tensor ownership to a copied holder, leaving meta tensors."""

        def to_meta(tensor, parameter=False):
            if id(tensor) not in memo:
                empty = torch.empty_like(tensor, device="meta")
                # for tracing
                memo[id(tensor)] = (
                    Parameter(empty, requires_grad=tensor.requires_grad)
                    if parameter or isinstance(tensor, Parameter)
                    else empty
                )
            return memo[id(tensor)]

        source = copy.copy(module)
        # Retain the source caches and remove dispatch from the live meta module.
        if isinstance(source._parameters, OffloadCache):
            remove_module_offload(module, onload_tensors=False)

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

    def apply_restructuring(self) -> None:
        """Restructure the meta model and associate replacements with real sources."""
        for restructure in self.restructures:
            self.model = restructure.apply(self.model, self._record_restructure)

    def _record_restructure(self, _name: str, module: Module, target: Module) -> None:
        # A previous parent replacement already owns any newly created children.
        if module not in self.mapping:
            return
        source = self.mapping[module]
        for old in module.modules():
            self.mapping.pop(old, None)
            self.restructured_modules.discard(old)
        self.mapping[target] = source
        self.restructured_modules.add(target)

    def configure_onload(self, device: torch.device) -> None:
        if self.source_root is not None:
            set_onload_device(self.source_root, device)

    def _release_source(self, source: Module, recurse: bool) -> None:
        retained = {
            module
            for original, _ in self.restore_sources.values()
            for module in original.modules()
        }
        for module in source.modules() if recurse else (source,):
            if module in retained:
                continue
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
            if target in self.restructured_modules:
                sources.update(source.named_modules(prefix=name))
            else:
                sources[name] = source
        kwargs = subgraph_onload_modules(sources)
        for name, target in pending.items():
            source = self.mapping.pop(target)
            restructured = target in self.restructured_modules
            self.restructured_modules.discard(target)
            remove_module_offload(target, onload_tensors=True)
            if not restructured:
                copy_module_state(target, source, device, recurse=False)
                self._materialize_added_children(target, device)
            else:

                def retain_for_restore(relative_name, original, replacement):
                    if restructure_requires_restore(replacement):
                        module_name = ".".join(
                            part for part in (name, relative_name) if part
                        )
                        live = target.get_submodule(relative_name)
                        settings = (
                            kwargs.get(module_name)
                            or kwargs.get(name)
                            or get_cache_init_kwargs(original)
                        )
                        self.restore_sources[live] = (original, settings)

                converted = source
                for restructure in self.restructures:
                    converted = restructure.apply(converted, retain_for_restore)
                copy_module_state(target, converted, device)
            self._release_source(source, recurse=restructured)
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
            # Restored sources execute forward again; copied instance forwards
            # must bind to the restored module rather than the old meta module.
            source.forward = source.forward.__func__.__get__(source)
            target.restore(source)
            if offload_sources:
                offload_module(source, **kwargs)
            self.model = replace_module(self.model, target, source)
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
        self.restructured_modules.clear()
        self.source_root = None
