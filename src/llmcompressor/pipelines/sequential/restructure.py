"""Model restructuring and materialization during calibration."""

from __future__ import annotations

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
    RESTRUCTURE_METADATA_ATTR,
    ModuleRestructure,
    copy_module_state,
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
    "restructure_requires_restore",
    "restore_restructured_modules",
]


def restructure_requires_restore(target: Module) -> bool:
    """Return whether a restructured target needs its source restored afterward."""
    # just this restructure for now, add moe repack later
    return isinstance(target, NormCalibrationModule)


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
            # each restructure target is a leaf node, don't nest them
            if any(
                fqn == root or fqn.startswith(f"{root}.") or root.startswith(f"{fqn}.")
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


def offload_original_modules(model: Module, onload_device: torch.device) -> None:
    """Configure offloading for the retained original modules."""
    for module in model.modules():
        if hasattr(module, RESTRUCTURE_METADATA_ATTR):
            original = getattr(module, RESTRUCTURE_METADATA_ATTR)["original"]
            set_onload_device(original, onload_device)


def onload_and_restructure_modules(
    modules: dict[str, Module],
    device: torch.device,
) -> dict[str, dict]:
    """Materialize pending restructures in the mapping and return offload kwargs."""
    pending: dict[str, Module] = {}
    for name, target in modules.items():
        metadata = getattr(target, RESTRUCTURE_METADATA_ATTR, None)
        if metadata is not None and not metadata["materialized"]:
            pending[name] = target

    source_modules = {
        name: getattr(target, RESTRUCTURE_METADATA_ATTR)["original"]
        for name, target in pending.items()
    }

    # onload original modules
    source_offload_kwargs = subgraph_onload_modules(source_modules, recurse=True)
    # transfer the offload kwargs
    target_offload_kwargs = {name: source_offload_kwargs[name] for name in pending}

    # iterate through targets and perform conversions
    for name, target in pending.items():
        metadata = getattr(target, RESTRUCTURE_METADATA_ATTR)
        source = metadata["original"]
        remove_dispatch(target, onload_tensors=True)
        converted = metadata["transform"].restructure(source, metadata["context"])
        copy_module_state(target, converted, device)
        metadata["materialized"] = True

        # Keep source metadata only for restructures that need restoration.
        if restructure_requires_restore(target):
            metadata["source_offload_kwargs"] = source_offload_kwargs[name]
        else:
            delattr(target, RESTRUCTURE_METADATA_ATTR)

    return target_offload_kwargs


def restructure_and_swap_in(model: Module) -> None:
    """Populate and offload targets for pipelines without subgraph support."""
    onload_device = get_main_device()
    set_onload_device(model, onload_device)
    offload_original_modules(model, onload_device)
    modules = {
        name: module
        for name, module in model.named_modules()
        if hasattr(module, RESTRUCTURE_METADATA_ATTR)
    }
    offload_kwargs = onload_and_restructure_modules(
        modules,
        onload_device,
    )
    subgraph_offload_modules(modules, offload_kwargs)


def restore_restructured_modules(
    model: Module,
    modules: dict[str, Module] | None = None,
    *,
    offload_sources: bool = True,
) -> None:
    """Restore temporary conversions in the model and supplied module mapping."""
    if modules is None:
        modules = dict(model.named_modules())

    for name, target in modules.items():
        if not hasattr(target, RESTRUCTURE_METADATA_ATTR):
            continue
        metadata = getattr(target, RESTRUCTURE_METADATA_ATTR)
        source = metadata["original"]
        if not metadata["materialized"]:
            model.set_submodule(metadata["fqn"], source)
            modules[name] = source
        elif restructure_requires_restore(target):
            source_kwargs = metadata.get("source_offload_kwargs")
            if source_kwargs is None:
                source_kwargs = get_cache_init_kwargs(source)
            remove_module_offload(target, onload_tensors=True)
            remove_module_offload(source, onload_tensors=True)
            target.restore(source)
            if offload_sources:
                offload_module(source, **source_kwargs)
            model.set_submodule(metadata["fqn"], source)
            modules[name] = source
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
