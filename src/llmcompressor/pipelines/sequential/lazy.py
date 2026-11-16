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
    "materialize_whole_model_meta_eagerly",
    "onload_and_convert_meta_modules",
    "offload_lazy_module_sources",
    "prepare_lazy_module_transforms",
    "prepare_whole_model_meta",
    "restore_lazy_module_transforms",
    "source_modules_for_subgraph",
    "transformed_fqns_for",
    "WholeModelMetaTransform",
]


def prepare_lazy_module_transforms(
    model: Module, transforms: Sequence[ModuleTransform]
) -> None:
    """Ask each model-wide transform to install and index its meta targets."""
    transformed_fqns: set[str] = set()
    for transform in transforms:
        transformed_fqns.update(transform.to_meta(model, transformed_fqns))


def prepare_whole_model_meta(
    source_model: Module, transforms: Sequence[ModuleTransform]
) -> Module:
    """Copy the complete model to meta, then restructure its meta copy."""
    meta_model = copy_module_to_meta(source_model)
    prepare_lazy_module_transforms(meta_model, transforms)

    # The transform discovery pass ran on the meta copy. Replace its retained
    # source references with the real model modules used to populate the copy.
    for transform in transforms:
        for fqn in list(transform.modules):
            transform.record_module(
                fqn,
                _get_submodule(source_model, fqn),
                _get_submodule(meta_model, fqn),
            )
    return meta_model


class WholeModelMetaTransform:
    """Hold the live source model beside a fully meta, restructured copy."""

    def __init__(
        self, source_model: Module, transforms: Sequence[ModuleTransform]
    ) -> None:
        self.source_model = source_model
        self.transforms = transforms
        self.model = prepare_whole_model_meta(source_model, transforms)


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


def onload_and_convert_meta_modules(
    transforms: Sequence[ModuleTransform],
    target_modules: Mapping[str, Module],
    offload_kwargs: dict[str, dict],
    device: torch.device,
    source_model: Module | None = None,
) -> None:
    """Materialize and onload the meta targets owned by a subgraph.

    Ordinary modules should already have been onloaded, with their cache settings
    recorded in ``offload_kwargs``. When ``source_model`` is supplied, matching
    ordinary modules are also materialized from it. Retained transform sources are
    onloaded here, used to populate their meta targets, and offloaded again. The
    resulting target modules are then onloaded through the regular offload helper.
    """
    if source_model is None:
        source_modules = source_modules_for_subgraph(transforms, target_modules)
    else:
        source_modules = _source_modules_for_meta_subgraph(
            source_model, transforms, target_modules
        )
    if not source_modules:
        return

    source_offload_kwargs = subgraph_onload_modules(source_modules)
    target_offload_kwargs = materialize_lazy_module_transforms(
        transforms,
        target_modules,
        source_modules,
        source_offload_kwargs,
        device,
    )
    if source_model is not None:
        target_offload_kwargs.update(
            _materialize_matching_meta_modules(
                target_modules,
                source_modules,
                transformed_fqns_for(transforms, target_modules),
                source_offload_kwargs,
                device,
            )
        )
    offload_kwargs.update(target_offload_kwargs)
    subgraph_offload_modules(source_modules, source_offload_kwargs)

    # Materialization creates normal offload caches on the targets. Reuse the
    # standard path to onload them for calibration and capture their cache settings.
    offload_kwargs.update(subgraph_onload_modules(target_modules))


def materialize_whole_model_meta_eagerly(
    model: Module,
    source_model: Module,
    transforms: Sequence[ModuleTransform],
    onload_device: torch.device,
) -> None:
    """Populate a complete meta copy before a pipeline without subgraph support."""
    offload_lazy_module_sources(source_model, transforms, onload_device)
    target_modules = dict(model.named_modules())
    source_modules = _source_modules_for_meta_subgraph(
        source_model, transforms, target_modules
    )
    source_offload_kwargs = subgraph_onload_modules(source_modules)
    target_offload_kwargs = materialize_lazy_module_transforms(
        transforms,
        target_modules,
        source_modules,
        source_offload_kwargs,
        onload_device,
    )
    target_offload_kwargs.update(
        _materialize_matching_meta_modules(
            target_modules,
            source_modules,
            transformed_fqns_for(transforms, target_modules),
            source_offload_kwargs,
            onload_device,
        )
    )
    subgraph_offload_modules(source_modules, source_offload_kwargs)


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
    *,
    recurse: bool = True,
) -> dict[str, dict]:
    source_state = _named_state(source, recurse=recurse)
    target_meta_state = _named_state(target, recurse=recurse)
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
        target_state = _named_state(module, recurse=recurse)
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

    materialize_meta_module(
        target, device=device, initializer=initialize, recurse=recurse
    )
    modules = target.modules() if recurse else (target,)
    for module in modules:
        offload_module(module, **offload_kwargs)
    if not recurse:
        return {"": offload_kwargs}
    return {relative_name: offload_kwargs for relative_name, _ in target.named_modules()}


def _named_state(module: Module, recurse: bool = True) -> dict[str, torch.Tensor]:
    state = dict(module.named_parameters(remove_duplicate=False, recurse=recurse))
    state.update(
        dict(module.named_buffers(remove_duplicate=False, recurse=recurse))
    )
    return state


def _source_modules_for_meta_subgraph(
    source_model: Module,
    transforms: Sequence[ModuleTransform],
    target_modules: Mapping[str, Module],
) -> dict[str, Module]:
    """Map unchanged modules by FQN and add original modules for transformed roots."""
    transformed_roots = transformed_fqns_for(transforms, target_modules)
    modules = {
        fqn: _get_submodule(source_model, fqn)
        for fqn in target_modules
        if not _fqn_is_transformed(fqn, transformed_roots)
    }
    modules.update(source_modules_for_subgraph(transforms, target_modules))
    return modules


def _materialize_matching_meta_modules(
    target_modules: Mapping[str, Module],
    source_modules: Mapping[str, Module],
    transformed_roots: set[str],
    source_offload_kwargs: Mapping[str, dict],
    device: torch.device,
) -> dict[str, dict]:
    """Copy unchanged modules into the whole-model meta snapshot by matching FQN."""
    offload_kwargs: dict[str, dict] = {}
    for fqn, target in target_modules.items():
        if _fqn_is_transformed(fqn, transformed_roots):
            continue
        source = source_modules.get(fqn)
        if source is None or not any(
            tensor is not None and tensor.device.type == "meta"
            for tensor in (*target._parameters.values(), *target._buffers.values())
        ):
            continue

        cache_kwargs = source_offload_kwargs.get(fqn)
        if cache_kwargs is None:
            cache_kwargs = get_cache_init_kwargs(source)
        relative_kwargs = _copy_module_state(
            target,
            source,
            device,
            cache_kwargs,
            recurse=False,
        )
        offload_kwargs.update(
            {_qualify_fqn(fqn, name): kwargs for name, kwargs in relative_kwargs.items()}
        )
    return offload_kwargs


def _fqn_is_transformed(fqn: str, roots: set[str]) -> bool:
    return any(
        fqn == root or not root or fqn.startswith(f"{root}.") for root in roots
    )


def _get_submodule(module: Module, fqn: str) -> Module:
    return module.get_submodule(fqn) if fqn else module
