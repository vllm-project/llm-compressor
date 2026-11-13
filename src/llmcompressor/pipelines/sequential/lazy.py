"""Model-wide transforms for lazy module materialization during calibration."""

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

from llmcompressor.modeling.module_transform import (
    ModuleTransform,
    fqn_overlaps,
    subgraph_owns_fqn,
)
from llmcompressor.modeling.moe.linearize import (
    MoELinearizationModuleTransform,
)
from llmcompressor.modeling.offset_norm import (
    NormCalibrationModule,
    get_norm_calibration_targets,
)
from llmcompressor.utils.pytorch.meta import (
    copy_module_to_meta,
    materialize_meta_module,
)

__all__ = [
    "ModuleTransform",
    "build_lazy_module_transforms",
    "materialize_lazy_module_transforms",
    "materialize_lazy_module_transforms_eagerly",
    "offload_lazy_module_sources",
    "prepare_lazy_module_transforms",
    "restore_lazy_module_transforms",
    "source_modules_for_subgraph",
    "transformed_fqns_for",
]


def prepare_lazy_module_transforms(
    model: Module, transforms: Sequence[ModuleTransform]
) -> None:
    """Ask each model-wide transform to install and index its meta targets."""
    transformed_fqns: set[str] = set()
    for transform in transforms:
        transformed_fqns.update(transform.to_meta(model, transformed_fqns))


def offload_lazy_module_sources(
    model: Module,
    transforms: Sequence[ModuleTransform],
    onload_device: torch.device,
) -> None:
    """Offload original modules while keeping meta targets installed in the model."""
    pending = []
    for transform in transforms:
        for fqn, (source, target) in transform.modules.items():
            if fqn not in transform.source_offloaded:
                pending.append((transform, fqn, source, target))
    if not pending:
        if not any(transform.modules for transform in transforms):
            set_onload_device(model, onload_device)
        return

    targets = {}
    for _transform, fqn, source, _target in pending:
        targets[fqn] = _get_submodule(model, fqn)
        model.set_submodule(fqn, source)

    try:
        set_onload_device(model, onload_device)
    finally:
        for _transform, fqn, _source, _target in pending:
            model.set_submodule(fqn, targets[fqn])

    for transform, fqn, source, _target in pending:
        transform.offload_kwargs[fqn] = get_cache_init_kwargs(source)
        transform.source_offloaded.add(fqn)


def source_modules_for_subgraph(
    transforms: Sequence[ModuleTransform],
    target_modules: Mapping[str, Module],
) -> dict[str, Module]:
    """Collect original modules held by transforms for this subgraph."""
    modules: dict[str, Module] = {}
    for transform in transforms:
        modules.update(transform.source_modules_for_subgraph(target_modules))
    return modules


def transformed_fqns_for(
    transforms: Sequence[ModuleTransform], target_modules: Mapping[str, Module]
) -> set[str]:
    """Return transformed roots owned by the given target modules."""
    return {
        fqn
        for transform in transforms
        for fqn in transform.modules
        if subgraph_owns_fqn(target_modules, fqn)
    }


def materialize_lazy_module_transforms(
    transforms: Sequence[ModuleTransform],
    target_modules: Mapping[str, Module],
    source_modules: Mapping[str, Module],
    source_offload_kwargs: Mapping[str, dict],
    device: torch.device,
) -> dict[str, dict]:
    """Populate the initialized targets owned by the current subgraph."""
    target_offload_kwargs: dict[str, dict] = {}
    for transform in transforms:
        target_offload_kwargs.update(
            transform.materialize(
                target_modules, source_modules, source_offload_kwargs, device
            )
        )

    return target_offload_kwargs


def materialize_lazy_module_transforms_eagerly(
    model: Module,
    transforms: Sequence[ModuleTransform],
    onload_device: torch.device,
) -> None:
    """Populate every target before running a pipeline without subgraph support."""
    if not any(transform.modules for transform in transforms):
        return

    offload_lazy_module_sources(model, transforms, onload_device)
    target_modules = dict(model.named_modules())
    for fqn in transformed_fqns_for(transforms, target_modules):
        target_subgraph_modules = {fqn: target_modules[fqn]}
        source_modules = source_modules_for_subgraph(
            transforms, target_subgraph_modules
        )
        source_offload_kwargs = subgraph_onload_modules(source_modules)
        try:
            materialize_lazy_module_transforms(
                transforms,
                target_subgraph_modules,
                source_modules,
                source_offload_kwargs,
                onload_device,
            )
        finally:
            subgraph_offload_modules(source_modules, source_offload_kwargs)


def restore_lazy_module_transforms(
    model: Module,
    transforms: Sequence[ModuleTransform],
    fqns: set[str] | None = None,
) -> None:
    """Commit or restore selected transforms and release their retained modules."""
    for transform in transforms:
        transform.restore(model, fqns)


class OffsetNormTransform(ModuleTransform):
    """Model-wide offset-norm conversion with per-subgraph reversion."""

    def __init__(self):
        super().__init__()
        self.config: Any = None
        self.class_names: dict[str, str] = {}

    def to_meta(self, model: Module, excluded_fqns: set[str]) -> set[str]:
        targets = get_norm_calibration_targets(model)
        if not targets:
            return set()

        self.config = getattr(model, "config", None)
        if self.config is None:
            raise ValueError("Offset norm transforms require a model config")

        transformed_fqns = set()
        for fqn, source, class_name in targets:
            if fqn_overlaps(fqn, excluded_fqns | transformed_fqns):
                continue
            source_meta = copy_module_to_meta(source)
            target = NormCalibrationModule.load_from_registry(
                class_name,
                original=source_meta,
                config=self.config,
            )
            self.record_module(fqn, source, target)
            self.class_names[fqn] = class_name
            model.set_submodule(fqn, target)
            transformed_fqns.add(fqn)
        return transformed_fqns

    def materialize_module(
        self,
        fqn: str,
        target: Module,
        source: Module,
        device: torch.device,
        offload_kwargs: dict,
    ) -> dict[str, dict]:
        transformed_source = NormCalibrationModule.load_from_registry(
            self.class_names[fqn],
            original=source,
            config=self.config,
        )
        return _copy_module_state(target, transformed_source, device, offload_kwargs)

    def restore_module(
        self,
        model: Module,
        fqn: str,
        source: Module,
        target: Module,
        materialized: bool,
    ) -> None:
        if not materialized:
            model.set_submodule(fqn, source)
            return
        if not isinstance(target, NormCalibrationModule):
            raise TypeError("Offset norm target must implement NormCalibrationModule")

        source_kwargs = self.offload_kwargs.get(fqn) or get_cache_init_kwargs(source)
        remove_module_offload(target, onload_tensors=True)
        remove_module_offload(source, onload_tensors=True)
        target.restore(source)
        offload_module(source, **source_kwargs)
        model.set_submodule(fqn, source)

    def forget_module(self, fqn: str) -> None:
        self.class_names.pop(fqn, None)
        super().forget_module(fqn)


class DecompressionTransform(ModuleTransform):
    """Model-wide decompression that materializes targets by subgraph."""

    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled = enabled

    def to_meta(self, model: Module, excluded_fqns: set[str]) -> set[str]:
        if not self.enabled:
            return set()

        transformed_fqns = set()
        for fqn, source in list(model.named_modules()):
            if not is_module_quantized(source) or fqn_overlaps(
                fqn, excluded_fqns | transformed_fqns
            ):
                continue
            target = copy_module_to_meta(source)
            for submodule in list(target.modules()):
                if is_module_quantized(submodule):
                    decompress_module(submodule, leave_decompressed=False)
            self.record_module(fqn, source, target)
            model.set_submodule(fqn, target)
            transformed_fqns.add(fqn)
        return transformed_fqns

    def materialize_module(
        self,
        _fqn: str,
        target: Module,
        source: Module,
        device: torch.device,
        offload_kwargs: dict,
    ) -> dict[str, dict]:
        for submodule in list(source.modules()):
            if is_module_quantized(submodule):
                decompress_module(submodule, leave_decompressed=False)
        return _copy_module_state(target, source, device, offload_kwargs)


def build_lazy_module_transforms(
    *,
    layerwise_decompression: bool = False,
) -> list[ModuleTransform]:
    """Build one model-wide transform for each kind of module conversion."""
    return [
        MoELinearizationModuleTransform(decompress=layerwise_decompression),
        OffsetNormTransform(),
        DecompressionTransform(enabled=layerwise_decompression),
    ]


def _copy_module_state(
    target: Module,
    source: Module,
    device: torch.device,
    offload_kwargs: dict,
) -> dict[str, dict]:
    source_state = _named_state(source)
    target_meta_state = _named_state(target)
    missing_source_names = set(source_state) - set(target_meta_state)
    if missing_source_names:
        raise ValueError(
            "Transformed target is missing source state entries: "
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
    for module in target.modules():
        offload_module(module, **offload_kwargs)
    return {
        relative_name: offload_kwargs
        for relative_name, _ in target.named_modules()
    }


def _named_state(module: Module) -> dict[str, torch.Tensor]:
    state = dict(module.named_parameters(remove_duplicate=False))
    state.update(dict(module.named_buffers(remove_duplicate=False)))
    return state


def _get_submodule(module: Module, fqn: str) -> Module:
    return module.get_submodule(fqn) if fqn else module
