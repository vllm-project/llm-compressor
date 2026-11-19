"""Model restructuring and materialization during calibration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from compressed_tensors.compressors import decompress_module
from compressed_tensors.offload import (
    get_cache_init_kwargs,
    offload_module,
    remove_dispatch,
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
    restructure_targets: Sequence[Module],
    device: torch.device,
) -> tuple[dict[str, Module], dict[str, dict]]:
    """Onload sources, materialize targets, and return their offload plan."""
    pending: list[tuple[Module, dict[str, Any]]] = []
    for target in restructure_targets:
        metadata = getattr(target, RESTRUCTURE_METADATA_ATTR)
        if not metadata["materialized"]:
            pending.append((target, metadata))

    source_modules = {
        metadata["fqn"]: metadata["original"] for _, metadata in pending
    }

    # onload original modules
    source_offload_kwargs = subgraph_onload_modules(source_modules, recurse=True)
    restructured_roots = {
        metadata["fqn"]: target for target, metadata in pending
    }

    # transfer the offload kwargs
    target_offload_kwargs = {
        fqn: source_offload_kwargs[fqn] for fqn in restructured_roots
    }

    # iterate through targets and perform conversions
    for target, metadata in pending:
        source = metadata["original"]
        remove_dispatch(target, onload_tensors=True)
        converted = metadata["transform"].restructure(source, metadata["context"])
        copy_module_state(target, converted, device)
        metadata["materialized"] = True

        # Offset-norm conversions retain their originals until calibration is complete
        # Delete all others
        if not isinstance(target, NormCalibrationModule):
            delattr(target, RESTRUCTURE_METADATA_ATTR)
        else:
            metadata["source_offload_kwargs"] = source_offload_kwargs[
                metadata["fqn"]
            ]

    return restructured_roots, target_offload_kwargs


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
    restructured_roots, target_offload_kwargs = onload_and_restructure_modules(
        restructure_targets,
        onload_device,
    )
    subgraph_offload_modules(restructured_roots, target_offload_kwargs)


def restore_restructured_modules(
    model: Module,
    targets: Sequence[Module] | None = None,
    *,
    offload_sources: bool = True,
) -> dict[str, Module]:
    """Restore temporary conversions and return original modules reinstated."""
    if targets is None:
        targets = [
            module
            for module in model.modules()
            if hasattr(module, RESTRUCTURE_METADATA_ATTR)
        ]

    restored_sources = {}
    for target in targets:
        if not hasattr(target, RESTRUCTURE_METADATA_ATTR):
            continue
        metadata = getattr(target, RESTRUCTURE_METADATA_ATTR)
        source = metadata["original"]
        if not metadata["materialized"]:
            model.set_submodule(metadata["fqn"], source)
        elif isinstance(target, NormCalibrationModule):
            source_kwargs = metadata.get("source_offload_kwargs")
            if source_kwargs is None:
                source_kwargs = get_cache_init_kwargs(source)
            remove_module_offload(target, onload_tensors=True)
            remove_module_offload(source, onload_tensors=True)
            target.restore(source)
            if offload_sources:
                offload_module(source, **source_kwargs)
            model.set_submodule(metadata["fqn"], source)
            restored_sources[metadata["fqn"]] = source
        delattr(target, RESTRUCTURE_METADATA_ATTR)

    return restored_sources


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
