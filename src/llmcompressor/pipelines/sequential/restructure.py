"""Model restructuring and materialization during calibration."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from compressed_tensors.compressors import decompress_module
from compressed_tensors.offload import (
    get_cache_init_kwargs,
    offload_module,
    set_onload_device,
)
from compressed_tensors.offload.module import (
    remove_module_offload,
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from compressed_tensors.quantization.utils import is_module_quantized
from torch.nn import Module

from llmcompressor.modeling.module_restructure import (
    ModuleRestructure,
    copy_module_state,
    fqn_overlaps,
    subgraph_owns_fqn,
)
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import (
    MoELinearizationRestructure,
    _experts_config,
)
from llmcompressor.modeling.offset_norm import (
    NormCalibrationModule,
    get_norm_calibration_targets,
)
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.pytorch.meta import copy_module_to_meta

__all__ = [
    "swap_in_meta_modules",
    "restructure_and_swap_in",
    "offload_original_modules",
    "onload_and_restructure_modules",
    "restore_restructured_modules",
    "restructured_fqns_for",
]


def swap_in_meta_modules(
    model: Module, *, layerwise_decompression: bool = False
) -> dict[str, Module]:
    """Install restructured meta copies and return original modules by FQN."""
    fqn_to_restucture_mods: dict[str, Module] = {}
    restructures = (
        MoELinearizationRestructure(decompress=layerwise_decompression),
        OffsetNormRestructure(),
        DecompressionRestructure(enabled=layerwise_decompression),
    )
    for restructure in restructures:
        for fqn, source, context in restructure.targets(model):
            if fqn_overlaps(fqn, set(fqn_to_restucture_mods)):
                continue
            target = restructure.restructure(copy_module_to_meta(source), context)
            fqn_to_restucture_mods[fqn] = source
            model.set_submodule(fqn, target)
    return fqn_to_restucture_mods


def offload_original_modules(
    fqn_to_restucture_mods: Mapping[str, Module],
    onload_device: torch.device,
) -> None:
    """Configure offloading for the retained original modules."""
    for module in fqn_to_restucture_mods.values():
        set_onload_device(module, onload_device)


def restructured_fqns_for(
    fqn_to_restucture_mods: Mapping[str, Module], target_modules: Mapping[str, Module]
) -> set[str]:
    """Return original roots owned by the given target modules."""
    return {
        fqn for fqn in fqn_to_restucture_mods if subgraph_owns_fqn(target_modules, fqn)
    }


def _restructure_source(
    model: Module, source: Module, target: Module, layerwise_decompression: bool
) -> Module:
    """Apply the same operation used to create the installed target."""
    if isinstance(target, LinearExperts2D):
        decompress = layerwise_decompression and any(
            is_module_quantized(child) for child in source.modules()
        )
        return MoELinearizationRestructure().restructure(
            source, (_experts_config(source, model), decompress)
        )
    if isinstance(target, NormCalibrationModule):
        return OffsetNormRestructure().restructure(
            source, (source.__class__.__name__, model.config)
        )
    return DecompressionRestructure(enabled=True).restructure(source, None)


def onload_and_restructure_modules(
    model: Module,
    fqn_to_restucture_mods: Mapping[str, Module],
    onloaded_modules: Mapping[str, Module],
    device: torch.device,
    layerwise_decompression: bool,
    *,
    offload_kwargs: dict[str, dict] | None = None,
    retain_targets: bool = True,
) -> None:
    """Onload originals, restructure them, and populate initialized targets."""
    source_modules = {
        (f"{root}.{name}" if name else root): module
        for root in fqn_to_restucture_mods
        for name, module in fqn_to_restucture_mods[root].named_modules()
    }
    root_kwargs = {
        root: get_cache_init_kwargs(source)
        for root, source in fqn_to_restucture_mods.items()
    }
    source_offload_kwargs = subgraph_onload_modules(source_modules)
    target_offload_kwargs: dict[str, dict] = {}
    try:
        for root, source in fqn_to_restucture_mods.items():
            target = _get_submodule(model, root)
            for module in target.modules():
                remove_module_offload(module, onload_tensors=True)
            converted = _restructure_source(
                model, source, target, layerwise_decompression
            )
            relative_kwargs = copy_module_state(
                target, converted, device, root_kwargs[root]
            )
            target_offload_kwargs.update(
                {
                    (f"{root}.{name}" if name else root): kwargs
                    for name, kwargs in relative_kwargs.items()
                }
            )
    finally:
        subgraph_offload_modules(source_modules, source_offload_kwargs)

    if offload_kwargs is not None:
        offload_kwargs.update(target_offload_kwargs)
    onloaded = set(onloaded_modules.values()) if retain_targets else set()
    for name, kwargs in target_offload_kwargs.items():
        module = _get_submodule(model, name)
        if module not in onloaded:
            offload_module(module, **kwargs)


def restructure_and_swap_in(
    model: Module,
    fqn_to_restucture_mods: Mapping[str, Module],
    layerwise_decompression: bool,
) -> None:
    """Populate and offload targets for pipelines without subgraph support."""
    onload_device = get_main_device()
    set_onload_device(model, onload_device)
    offload_original_modules(fqn_to_restucture_mods, onload_device)
    for fqn, source in fqn_to_restucture_mods.items():
        onload_and_restructure_modules(
            model,
            {fqn: source},
            {fqn: _get_submodule(model, fqn)},
            onload_device,
            layerwise_decompression,
            retain_targets=False,
        )


def restore_restructured_modules(
    model: Module,
    fqn_to_restucture_mods: dict[str, Module],
    materialized: set[str] | None = None,
    fqns: set[str] | None = None,
) -> None:
    """Restore conversions and release originals; default to all materialized."""
    for fqn in list(fqn_to_restucture_mods):
        if fqns is not None and fqn not in fqns:
            continue
        source = fqn_to_restucture_mods[fqn]
        target = _get_submodule(model, fqn)
        if materialized is not None and fqn not in materialized:
            model.set_submodule(fqn, source)
        elif isinstance(target, NormCalibrationModule):
            source_kwargs = get_cache_init_kwargs(source)
            remove_module_offload(target, onload_tensors=True)
            remove_module_offload(source, onload_tensors=True)
            target.restore(source)
            offload_module(source, **source_kwargs)
            model.set_submodule(fqn, source)
        fqn_to_restucture_mods.pop(fqn)
        if materialized is not None:
            materialized.discard(fqn)


class OffsetNormRestructure(ModuleRestructure):
    """Model-wide offset-norm conversion with per-subgraph reversion."""

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


class DecompressionRestructure(ModuleRestructure):
    """Model-wide decompression that materializes targets by subgraph."""

    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled = enabled

    def restructure(self, module: Module, context) -> Module:
        for submodule in list(module.modules()):
            if is_module_quantized(submodule):
                decompress_module(submodule, leave_decompressed=False)
        return module

    def targets(self, model: Module):
        if not self.enabled:
            return

        for fqn, source in list(model.named_modules()):
            if is_module_quantized(source):
                yield fqn, source, None


def _get_submodule(module: Module, fqn: str) -> Module:
    return module.get_submodule(fqn) if fqn else module
