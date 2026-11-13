"""Module-scoped meta replacements for sequential calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

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
    "release_lazy_module_sources",
    "restore_lazy_module_transforms",
    "source_modules_for",
    "transformed_fqns_for",
]


class ModuleTransform(Protocol):
    """Stateful, module-scoped transform with a two-phase materialization."""

    fqn: str
    source: Module | None
    target: Module | None
    offload_kwargs: dict
    source_offloaded: bool
    materialized: bool
    restored: bool
    # Some temporary forms can be restored after their final subgraph use. Others
    # must remain installed until all modifiers receive Calibration_End.
    restore_after_subgraph: bool

    def to_meta(self, source: Module, model: Module) -> Module: ...

    def materialize(
        self,
        target: Module,
        source: Module,
        device: torch.device,
        offload_kwargs: dict,
    ) -> dict[str, dict]: ...

    def restore(self, model: Module) -> None: ...


def prepare_lazy_module_transforms(
    model: Module, transforms: Sequence[ModuleTransform]
) -> None:
    """Install all meta targets before modifiers initialize against the model."""
    _validate_transform_paths(transforms)
    for transform in transforms:
        source = _get_submodule(model, transform.fqn)
        transform.source = source
        target = transform.to_meta(source, model)
        transform.target = target
        model.set_submodule(transform.fqn, target)


def offload_lazy_module_sources(
    model: Module,
    transforms: Sequence[ModuleTransform],
    onload_device: torch.device,
) -> None:
    """Configure model and retained-source offloading after modifier initialization."""
    if not transforms:
        set_onload_device(model, onload_device)
        return

    pending = [
        transform
        for transform in transforms
        if transform.source is not None and not transform.source_offloaded
    ]
    if not pending:
        return

    targets = {}
    for transform in pending:
        if transform.source is None:
            raise ValueError(f"Missing source module for {transform.fqn}")
        targets[transform.fqn] = _get_submodule(model, transform.fqn)
        model.set_submodule(transform.fqn, transform.source)

    try:
        set_onload_device(model, onload_device)
    finally:
        for transform in pending:
            target = targets[transform.fqn]
            transform.target = target
            model.set_submodule(transform.fqn, target)

    for transform in pending:
        if transform.source is None:
            raise ValueError(f"Missing source module for {transform.fqn}")
        transform.offload_kwargs = get_cache_init_kwargs(transform.source)
        transform.source_offloaded = True


def source_modules_for(
    transforms: Sequence[ModuleTransform],
    target_modules: Mapping[str, Module],
) -> dict[str, Module]:
    """Find retained source modules for transforms owned by a subgraph."""
    modules = {}
    for transform in transforms:
        if (
            transform.source is None
            or transform.materialized
            or transform.restored
            or not _subgraph_owns_fqn(target_modules, transform.fqn)
        ):
            continue
        for relative_name, module in transform.source.named_modules():
            name = (
                f"{transform.fqn}.{relative_name}" if relative_name else transform.fqn
            )
            modules[name] = module
    return modules


def transformed_fqns_for(
    transforms: Sequence[ModuleTransform], target_modules: Mapping[str, Module]
) -> set[str]:
    """Return transformed roots owned by the given target modules."""
    return {
        transform.fqn
        for transform in transforms
        if _subgraph_owns_fqn(target_modules, transform.fqn)
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
        if (
            transform.materialized
            or transform.restored
            or not _subgraph_owns_fqn(target_modules, transform.fqn)
        ):
            continue

        if transform.target is None:
            raise ValueError(f"Missing target module for {transform.fqn}")
        source = source_modules.get(transform.fqn)
        if source is None:
            raise ValueError(f"Missing source module for {transform.fqn}")
        offload_kwargs = source_offload_kwargs.get(
            transform.fqn, transform.offload_kwargs
        )
        relative_kwargs = transform.materialize(
            transform.target, source, device, offload_kwargs
        )
        transform.materialized = True
        for relative_name, kwargs in relative_kwargs.items():
            target_name = (
                f"{transform.fqn}.{relative_name}"
                if relative_name
                else transform.fqn
            )
            target_offload_kwargs[target_name] = kwargs

    return target_offload_kwargs


def materialize_lazy_module_transforms_eagerly(
    model: Module,
    transforms: Sequence[ModuleTransform],
    onload_device: torch.device,
) -> None:
    """Populate every target before running a pipeline without subgraph support."""
    if not transforms:
        return

    offload_lazy_module_sources(model, transforms, onload_device)
    for transform in transforms:
        if transform.materialized or transform.restored:
            continue

        if transform.target is None:
            raise ValueError(f"Missing target module for {transform.fqn}")
        target_modules = {transform.fqn: transform.target}
        source_modules = source_modules_for([transform], target_modules)
        source_offload_kwargs = subgraph_onload_modules(source_modules)
        try:
            materialize_lazy_module_transforms(
                [transform],
                target_modules,
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
        if transform.restored or (fqns is not None and transform.fqn not in fqns):
            continue
        transform.restore(model)
        transform.source = None
        transform.target = None
        transform.restored = True


def release_lazy_module_sources(
    transforms: Sequence[ModuleTransform], fqns: set[str]
) -> None:
    """Drop source modules after their final materialization use."""
    for transform in transforms:
        if transform.fqn in fqns:
            transform.source = None


class OffsetNormTransform:
    """Replace an offset norm with its meta calibration form, then restore it."""

    def __init__(self, fqn: str, class_name: str):
        self.fqn = fqn
        self.class_name = class_name
        self.config: Any = None
        self.source: Module | None = None
        self.target: Module | None = None
        self.offload_kwargs: dict = {}
        self.source_offloaded = False
        self.materialized = False
        self.restored = False
        self.restore_after_subgraph = True

    def to_meta(self, source: Module, model: Module) -> Module:
        self.config = getattr(model, "config", None)
        if self.config is None:
            raise ValueError("Offset norm transforms require a model config")
        source_meta = copy_module_to_meta(source)
        return NormCalibrationModule.load_from_registry(
            self.class_name,
            original=source_meta,
            config=self.config,
        )

    def materialize(
        self,
        target: Module,
        source: Module,
        device: torch.device,
        offload_kwargs: dict,
    ) -> dict[str, dict]:
        transformed_source = NormCalibrationModule.load_from_registry(
            self.class_name,
            original=source,
            config=self.config,
        )
        return _copy_module_state(target, transformed_source, device, offload_kwargs)

    def restore(self, model: Module) -> None:
        source, target = self.source, self.target
        if source is None:
            return
        if not self.materialized:
            model.set_submodule(self.fqn, source)
            return
        if not isinstance(target, NormCalibrationModule):
            raise TypeError("Offset norm target must implement NormCalibrationModule")

        source_kwargs = self.offload_kwargs or get_cache_init_kwargs(source)
        remove_module_offload(target, onload_tensors=True)
        remove_module_offload(source, onload_tensors=True)
        target.restore(source)
        offload_module(source, **source_kwargs)
        model.set_submodule(self.fqn, source)


class DecompressionTransform:
    """Create a decompressed meta structure and fill it from an onloaded module."""

    def __init__(self, fqn: str):
        self.fqn = fqn
        self.source: Module | None = None
        self.target: Module | None = None
        self.offload_kwargs: dict = {}
        self.source_offloaded = False
        self.materialized = False
        self.restored = False
        self.restore_after_subgraph = False

    def to_meta(self, source: Module, model: Module) -> Module:
        target = copy_module_to_meta(source)
        for submodule in list(target.modules()):
            if is_module_quantized(submodule):
                decompress_module(submodule, leave_decompressed=False)
        return target

    def materialize(
        self,
        target: Module,
        source: Module,
        device: torch.device,
        offload_kwargs: dict,
    ) -> dict[str, dict]:
        for submodule in list(source.modules()):
            if is_module_quantized(submodule):
                decompress_module(submodule, leave_decompressed=False)
        return _copy_module_state(target, source, device, offload_kwargs)

    def restore(self, model: Module) -> None:
        # The decompressed target is the calibrated model. Restore the source only
        # when this subgraph never materialized the target.
        if not self.materialized and self.source is not None:
            model.set_submodule(self.fqn, self.source)


def build_lazy_module_transforms(
    model: Module,
    *,
    layerwise_decompression: bool = False,
) -> list[ModuleTransform]:
    """Build one stateful transform for each module path that must change."""
    from llmcompressor.modeling.moe.linearize import (
        get_non_linearized_moes,
    )

    transforms: list[ModuleTransform] = []
    moe_modules = get_non_linearized_moes(model)
    transforms.extend(
        MoELinearizationModuleTransform(
            fqn,
            decompress=layerwise_decompression
            and any(is_module_quantized(child) for child in module.modules()),
        )
        for fqn, module in moe_modules
    )
    moe_roots = tuple(transform.fqn for transform in transforms)
    for fqn, _module, class_name in get_norm_calibration_targets(model):
        if any(fqn.startswith(f"{moe_fqn}.") for moe_fqn in moe_roots):
            continue
        transforms.append(OffsetNormTransform(fqn, class_name))

    if layerwise_decompression:
        covered_roots = tuple(transform.fqn for transform in transforms)
        for fqn, module in model.named_modules():
            if not is_module_quantized(module):
                continue
            if any(fqn == root or fqn.startswith(f"{root}.") for root in covered_roots):
                continue
            if any(root.startswith(f"{fqn}.") for root in covered_roots):
                continue
            transforms.append(DecompressionTransform(fqn))
            covered_roots += (fqn,)

    return transforms


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


def _subgraph_owns_fqn(target_modules: Mapping[str, Module], fqn: str) -> bool:
    return fqn in target_modules or any(
        name.startswith(f"{fqn}.") for name in target_modules
    )


def _get_submodule(module: Module, fqn: str) -> Module:
    return module.get_submodule(fqn) if fqn else module


def _validate_transform_paths(transforms: Sequence[ModuleTransform]) -> None:
    paths = sorted(transform.fqn for transform in transforms)
    for index, path in enumerate(paths):
        if any(
            other == path or other.startswith(f"{path}.")
            for other in paths[index + 1 :]
        ):
            raise ValueError(
                f"Lazy transformations cannot overlap: {path!r} is duplicated or "
                "contains another transformed module"
            )
