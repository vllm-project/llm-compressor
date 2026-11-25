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

__all__ = ["RestructureHandler"]


class RestructureHandler:
    """Retain real sources while the live model initializes and traces on meta.

    ``mapping`` maps live targets to their source, ordered restructures, and
    materialization status. Temporary restructures retain entries until restoration.
    """

    def __init__(self, model: Module, *, layerwise_decompression: bool = False):
        self.model = model
        self.mapping: dict[Module, dict] = {}
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
        self.mapping[module] = {
            "source": source,
            "restructures": [],
            "materialized": False,
        }
        return module

    def convert_model_to_meta(self) -> None:
        """Delay registered parameters/buffers, leaving live module attributes alone."""
        nodes = list(self.model.modules())
        memo: dict[int, torch.Tensor] = {}
        for module in nodes:
            self.delay_loading(module, memo)

    def apply_restructuring(self) -> None:
        """Restructure the meta model and associate replacements with real sources."""
        for restructure in self.restructures:
            self.model = restructure.apply(self.model, self.mapping)

    def restore_restructured_modules(
        self, modules: dict[str, Module] | None = None
    ) -> None:
        targets = (
            self.mapping
            if modules is None
            else {
                target: self.mapping[target]
                for target in modules.values()
                if target in self.mapping
            }
        )
        targets = {
            target: entry for target, entry in targets.items() if entry["materialized"]
        }
        for restructure in reversed(self.restructures):
            selected = {
                target: entry
                for target, entry in targets.items()
                if restructure in entry["restructures"]
            }
            self.model = restructure.restore(self.model, selected, modules)
        for target in targets:
            del self.mapping[target]
