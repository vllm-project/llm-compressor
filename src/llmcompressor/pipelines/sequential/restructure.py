"""Model restructuring and materialization during calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
    RESTRUCTURE_METADATA_ATTR,
)
from llmcompressor.modeling.moe.linearize import MoELinearizationRestructure
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
]


def swap_in_meta_modules(
    model: Module, *, layerwise_decompression: bool = False
) -> None:
    """Install restructured meta copies, recording each original on its target."""
    restructured_roots: set[str] = set()
    restructures = (
        MoELinearizationRestructure(decompress=layerwise_decompression),
        OffsetNormRestructure(),
        DecompressionRestructure(enabled=layerwise_decompression),
    )
    for restructure in restructures:
        for fqn, source, context in restructure.targets(model):
            if any(
                fqn == root
                or fqn.startswith(f"{root}.")
                or root.startswith(f"{fqn}.")
                for root in restructured_roots
            ):
                continue
            target = restructure.restructure(copy_module_to_meta(source), context)
            # Keep the source in a plain dict, outside PyTorch's child registry.
            setattr(
                target,
                RESTRUCTURE_METADATA_ATTR,
                {
                    "original": source,
                    "transform": restructure,
                    "context": context,
                    "fqn": fqn,
                    "materialized": False,
                },
            )
            restructured_roots.add(fqn)
            model.set_submodule(fqn, target)


def offload_original_modules(
    model: Module, onload_device: torch.device
) -> None:
    """Configure offloading for the retained original modules."""
    for module in model.modules():
        if hasattr(module, RESTRUCTURE_METADATA_ATTR):
            original = getattr(module, RESTRUCTURE_METADATA_ATTR)["original"]
            set_onload_device(original, onload_device)


def onload_and_restructure_modules(
    model: Module,
    restructure_targets: Sequence[Module],
    onloaded_modules: Mapping[str, Module],
    device: torch.device,
    *,
    offload_kwargs: dict[str, dict] | None = None,
    retain_targets: bool = True,
) -> None:
    """Onload originals, restructure them, and populate initialized targets."""
    pending: list[tuple[Module, dict[str, Any]]] = []
    for target in restructure_targets:
        if not hasattr(target, RESTRUCTURE_METADATA_ATTR):
            continue
        metadata = getattr(target, RESTRUCTURE_METADATA_ATTR)
        if not metadata["materialized"]:
            pending.append((target, metadata))
    source_modules = {
        (
            f"{metadata['fqn']}.{name}" if name else metadata["fqn"]
        ): module
        for _, metadata in pending
        for name, module in metadata["original"].named_modules()
    }
    root_kwargs = {
        metadata["fqn"]: get_cache_init_kwargs(metadata["original"])
        for _, metadata in pending
    }
    source_offload_kwargs = subgraph_onload_modules(source_modules)
    target_offload_kwargs: dict[str, dict] = {}
    try:
        for target, metadata in pending:
            source = metadata["original"]
            fqn = metadata["fqn"]
            for module in target.modules():
                remove_module_offload(module, onload_tensors=True)
            converted = metadata["transform"].restructure(source, metadata["context"])
            relative_kwargs = copy_module_state(
                target, converted, device, root_kwargs[fqn]
            )
            target_offload_kwargs.update(
                {
                    (f"{fqn}.{name}" if name else fqn): kwargs
                    for name, kwargs in relative_kwargs.items()
                }
            )
            metadata["materialized"] = True
            if not isinstance(target, NormCalibrationModule):
                delattr(target, RESTRUCTURE_METADATA_ATTR)
    finally:
        subgraph_offload_modules(source_modules, source_offload_kwargs)

    if offload_kwargs is not None:
        offload_kwargs.update(target_offload_kwargs)
    onloaded = set(onloaded_modules.values()) if retain_targets else set()
    for name, kwargs in target_offload_kwargs.items():
        module = _get_submodule(model, name)
        if module not in onloaded:
            offload_module(module, **kwargs)


def restructure_and_swap_in(model: Module) -> None:
    """Populate and offload targets for pipelines without subgraph support."""
    onload_device = get_main_device()
    set_onload_device(model, onload_device)
    offload_original_modules(model, onload_device)
    restructure_targets = [
        module
        for module in model.modules()
        if hasattr(module, RESTRUCTURE_METADATA_ATTR)
    ]
    onload_and_restructure_modules(
        model,
        restructure_targets,
        dict(model.named_modules()),
        onload_device,
        retain_targets=False,
    )


def restore_restructured_modules(
    model: Module,
    targets: Sequence[Module] | None = None,
) -> None:
    """Restore temporary conversions and release retained originals."""
    if targets is None:
        targets = [
            module
            for module in model.modules()
            if hasattr(module, RESTRUCTURE_METADATA_ATTR)
        ]

    for target in targets:
        if not hasattr(target, RESTRUCTURE_METADATA_ATTR):
            continue
        metadata = getattr(target, RESTRUCTURE_METADATA_ATTR)
        source = metadata["original"]
        if not metadata["materialized"]:
            model.set_submodule(metadata["fqn"], source)
        elif isinstance(target, NormCalibrationModule):
            source_kwargs = get_cache_init_kwargs(source)
            remove_module_offload(target, onload_tensors=True)
            remove_module_offload(source, onload_tensors=True)
            target.restore(source)
            offload_module(source, **source_kwargs)
            model.set_submodule(metadata["fqn"], source)
        delattr(target, RESTRUCTURE_METADATA_ATTR)


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
