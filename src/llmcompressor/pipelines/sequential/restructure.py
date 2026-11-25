"""Prototype whole-model meta initialization and delayed restructuring."""

from __future__ import annotations

import copy

import torch
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import remove_module_offload
from torch.nn import Module, Parameter

from llmcompressor.modeling.decompression import DecompressionRestructure
from llmcompressor.modeling.moe.linearize import MoELinearizationRestructure
from llmcompressor.modeling.offset_norm import OffsetNormRestructure
from llmcompressor.pipelines.sequential.offload import (
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from llmcompressor.utils.dev import get_main_device

__all__ = ["RestructureHandler"]


class RestructureHandler:
    """Retain real sources while the live model initializes and traces on meta.

    ``mapping`` maps current model modules to their temporary source holders.
    Each restructure owns its affected modules and any replacement-specific work.
    """

    def __init__(self, model: Module, *, layerwise_decompression: bool = False):
        self.model = model
        self.mapping: dict[Module, Module] = {}
        self.tensor_aliases: dict[torch.Tensor, list[tuple[dict, str]]] = {}
        self.restructures = (
            MoELinearizationRestructure(),
            DecompressionRestructure(enabled=layerwise_decompression),
            OffsetNormRestructure(),
        )

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
        # Source holders must not share mutable hook registries with live targets.
        for name, value in vars(source).items():
            if "hook" in name and isinstance(value, dict):
                source.__dict__[name] = value.copy()
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
        self.mapping[module] = source
        return module

    def convert_model_to_meta(self) -> None:
        """Delay registered parameters/buffers, leaving live module attributes alone."""
        nodes = list(self.model.modules())
        memo: dict[int, torch.Tensor] = {}
        for module in nodes:
            self.delay_loading(module, memo)
        for module in nodes:
            self.mapping[module]._modules = {
                name: self.mapping[child] if child is not None else None
                for name, child in module._modules.items()
            }

    def apply_restructuring(self) -> None:
        """Restructure the meta model and associate replacements with real sources."""
        for restructure in self.restructures:
            self.model = restructure.apply(self.model, self.mapping)
        # Replacing a placeholder must reconnect shared registered state as well.
        aliases = {}
        for module in self.model.modules():
            for state in (module._parameters, module._buffers):
                for name, tensor in state.items():
                    if tensor is not None:
                        aliases.setdefault(tensor, []).append((state, name))
        self.tensor_aliases = {
            tensor: entries for tensor, entries in aliases.items() if len(entries) > 1
        }

    def restore_restructured_modules(
        self, modules: dict[str, Module] | None = None
    ) -> None:
        for restructure in reversed(self.restructures):
            self.model = restructure.restore(self.model, modules)

    def finish(self) -> None:
        """Restore a complete usable model, including state outside traced graphs."""
        device = get_main_device()
        restore_targets = {
            target
            for restructure in self.restructures
            for target in restructure.mapping.values()
        }
        modules = {
            name: module
            for name, module in self.model.named_modules()
            if module in self.mapping or module in restore_targets
        }
        kwargs = subgraph_onload_modules(modules, self, device)
        self.restore_restructured_modules(modules)
        # Include materialized descendants while leaving existing caches untouched.
        modules = {
            name: module
            for name, module in self.model.named_modules()
            if not isinstance(module._parameters, OffloadCache)
        }
        subgraph_offload_modules(modules, kwargs)
        self.mapping.clear()
        self.tensor_aliases.clear()
