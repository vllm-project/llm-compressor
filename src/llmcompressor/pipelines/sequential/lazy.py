"""Module-scoped meta replacements for sequential calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from compressed_tensors.compressors import decompress_module
from compressed_tensors.offload import (
    get_cache_init_kwargs,
    offload_module,
    set_onload_device,
)
from compressed_tensors.offload.module import remove_module_offload
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

__all__ = ["LazyModuleSet", "build_lazy_module_transforms"]


class ModuleTransform(Protocol):
    """Convert one source module to a meta target and materialize it later."""

    def to_meta(self, source: Module, model: Module) -> Module: ...

    def materialize(
        self,
        target: Module,
        source: Module,
        device: torch.device,
        offload_kwargs: dict,
    ) -> dict[str, dict]: ...


@dataclass
class LazyModuleRecord:
    source: Module | None
    target: Module
    transform: ModuleTransform
    offload_kwargs: dict
    materialized: bool = False


class LazyModuleSet:
    """Keep real source modules aside and install only transformed meta modules."""

    def __init__(self, model: Module, records: dict[str, LazyModuleRecord]):
        self.model = model
        self.records = records
        self._sources_offloaded = False

    @classmethod
    def prepare(
        cls,
        model: Module,
        transforms: Mapping[str, ModuleTransform],
    ) -> LazyModuleSet:
        """Replace selected FQNs with meta trees while retaining real sources."""
        _validate_transform_paths(transforms)

        records = {}
        try:
            for fqn, transform in transforms.items():
                source = _get_submodule(model, fqn)
                target = transform.to_meta(source, model)
                model.set_submodule(fqn, target)
                records[fqn] = LazyModuleRecord(
                    source=source,
                    target=target,
                    transform=transform,
                    offload_kwargs={},
                )
        except BaseException:
            # A failed transform must not leave an earlier FQN replaced by a meta
            # module whose real source is about to go out of scope.
            for fqn, record in reversed(list(records.items())):
                if record.source is not None:
                    model.set_submodule(fqn, record.source)
            raise

        return cls(model, records)

    def offload_sources(self, onload_device: torch.device) -> None:
        """Offload real sources after modifiers have initialized on the meta model."""
        if self._sources_offloaded:
            return

        # set_onload_device must see the real source modules, and should run after
        # modifier initialization so initialization does not trigger model-wide
        # onloads merely by inspecting parameters.
        targets = {}
        for fqn, record in self.records.items():
            if record.source is None:
                raise ValueError(f"Missing source module for lazy transform {fqn}")
            targets[fqn] = self.model.get_submodule(fqn)
            self.model.set_submodule(fqn, record.source)

        set_onload_device(self.model, onload_device)

        for fqn, record in self.records.items():
            source = record.source
            if source is None:
                raise ValueError(f"Missing source module for lazy transform {fqn}")
            record.offload_kwargs = get_cache_init_kwargs(source)
            record.target = targets[fqn]
            self.model.set_submodule(fqn, targets[fqn])

        self._sources_offloaded = True

    @property
    def fqns(self) -> set[str]:
        return set(self.records)

    def source_modules_for(
        self, target_modules: Mapping[str, Module]
    ) -> dict[str, Module]:
        """Find source modules for transformed targets owned by a subgraph."""
        modules = {}
        for fqn in self.transformed_fqns_for(target_modules):
            record = self.records[fqn]
            if record.materialized or record.source is None:
                continue
            for relative_name, module in record.source.named_modules():
                name = f"{fqn}.{relative_name}" if relative_name else fqn
                modules[name] = module
        return modules

    def transformed_fqns_for(self, target_modules: Mapping[str, Module]) -> set[str]:
        """Return transformed roots owned by the given target modules."""
        return {
            fqn
            for fqn in self.records
            if _subgraph_owns_fqn(target_modules, fqn)
        }

    def release_sources(self, fqns: set[str]) -> None:
        """Drop sources after their targets finish calibration and need no restore."""
        for fqn in fqns:
            record = self.records[fqn]
            if getattr(record.transform, "finalize", None) is None:
                record.source = None

    def materialize_subgraph(
        self,
        target_modules: Mapping[str, Module],
        source_modules: Mapping[str, Module],
        source_offload_kwargs: Mapping[str, dict],
        device: torch.device,
    ) -> dict[str, dict]:
        """Fill each selected target in place and attach its source offload policy."""
        target_offload_kwargs: dict[str, dict] = {}
        for fqn, record in self.records.items():
            if not _subgraph_owns_fqn(target_modules, fqn) or record.materialized:
                continue

            source = source_modules.get(fqn)
            if source is None:
                raise ValueError(f"Missing source module for transformed target {fqn}")
            offload_kwargs = source_offload_kwargs.get(fqn, record.offload_kwargs)
            relative_kwargs = record.transform.materialize(
                record.target,
                source,
                device,
                offload_kwargs,
            )
            for relative_name, kwargs in relative_kwargs.items():
                target_name = (
                    f"{fqn}.{relative_name}" if relative_name else fqn
                )
                target_offload_kwargs[target_name] = kwargs
            record.materialized = True

        return target_offload_kwargs

    def finalize(self) -> None:
        """Restore transforms that are calibration-only, such as offset norms."""
        for fqn, record in self.records.items():
            finalize = getattr(record.transform, "finalize", None)
            if finalize is not None:
                finalize(fqn, self.model, record)
            elif not record.materialized and record.source is not None:
                self.model.set_submodule(fqn, record.source)
        # No transformed target needs its detached source after a successful run.
        # Offset norm finalizers reattach the source to the model before this point.
        for record in self.records.values():
            record.source = None

    def abort(self) -> None:
        """Restore sources still held if initialization or calibration fails.

        Transforms whose calibration completed have already committed their targets
        and released their sources. In-flight and not-yet-used transforms are restored.
        """
        for fqn, record in reversed(list(self.records.items())):
            source = record.source
            if source is None:
                continue
            for submodule in record.target.modules():
                remove_module_offload(submodule, onload_tensors=False)
            self.model.set_submodule(fqn, source)


class OffsetNormTransform:
    """Replace an offset norm with its calibration form on meta."""

    def __init__(self, class_name: str):
        self.class_name = class_name
        self.config: Any = None

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

    def finalize(self, fqn: str, model: Module, record: LazyModuleRecord) -> None:
        source = record.source
        if source is None:
            raise ValueError(f"Offset norm source for {fqn} has already been released")
        if not record.materialized:
            model.set_submodule(fqn, source)
            return

        source_kwargs = get_cache_init_kwargs(source)
        remove_module_offload(record.target, onload_tensors=True)
        remove_module_offload(source, onload_tensors=True)
        if not isinstance(record.target, NormCalibrationModule):
            raise TypeError("Offset norm target must implement NormCalibrationModule")
        record.target.restore(source)
        offload_module(source, **source_kwargs)
        model.set_submodule(fqn, source)


class DecompressionTransform:
    """Create a decompressed meta structure and fill it from an onloaded module."""

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


def build_lazy_module_transforms(
    model: Module,
    *,
    layerwise_decompression: bool = False,
) -> dict[str, ModuleTransform]:
    """List the module paths whose structures change for calibration."""
    from llmcompressor.modeling.moe.linearize import (
        get_non_linearized_moes,
    )

    moe_modules = get_non_linearized_moes(model)
    transforms: dict[str, ModuleTransform] = {
        fqn: MoELinearizationModuleTransform(
            decompress=layerwise_decompression
            and any(is_module_quantized(child) for child in module.modules())
        )
        for fqn, module in moe_modules
    }
    moe_roots = tuple(transforms)
    for fqn, _module, class_name in get_norm_calibration_targets(model):
        if any(fqn.startswith(f"{moe_fqn}.") for moe_fqn in moe_roots):
            continue
        transforms[fqn] = OffsetNormTransform(class_name)

    if layerwise_decompression:
        covered_roots = tuple(transforms)
        for fqn, module in model.named_modules():
            if not is_module_quantized(module):
                continue
            if any(fqn == root or fqn.startswith(f"{root}.") for root in covered_roots):
                continue
            if any(root.startswith(f"{fqn}.") for root in covered_roots):
                continue
            transforms[fqn] = DecompressionTransform()
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


def _validate_transform_paths(transforms: Mapping[str, ModuleTransform]) -> None:
    paths = sorted(transforms)
    for index, path in enumerate(paths):
        if any(other.startswith(f"{path}.") for other in paths[index + 1 :]):
            raise ValueError(
                f"Lazy transformations cannot overlap: {path!r} contains another "
                "transformed module"
            )
