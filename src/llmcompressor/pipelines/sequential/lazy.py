"""Whole-model meta snapshot and subgraph materialization prototype."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from compressed_tensors.compressors import decompress_module
from compressed_tensors.offload import (
    get_cache_init_kwargs,
    offload_module,
    set_onload_device,
)
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.module import (
    remove_module_offload,
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from compressed_tensors.quantization.utils import is_module_quantized
from torch.nn import Module
from transformers import PreTrainedModel

from llmcompressor.modeling.moe.linearize import (
    get_non_linearized_moes,
    linearize_moe_module,
)
from llmcompressor.modeling.offset_norm import (
    NormCalibrationModule,
    get_norm_calibration_targets,
)
from llmcompressor.utils.pytorch.meta import (
    copy_module_to_meta,
    materialize_meta_module,
)

__all__ = ["MetaOperation", "WholeModelMetaTransform", "build_meta_operations"]


MetaApply = Callable[[Module, set[str], dict[str, Any]], set[str]]
SourceApply = Callable[[Module, set[str], dict[str, Any]], None]
RestoreApply = Callable[[Module, Module, set[str], dict[str, Any]], None]


@dataclass
class MetaOperation:
    """Callbacks that apply one restructuring to meta and loaded source modules."""

    name: str
    restructure_meta: MetaApply
    restructure_source: SourceApply
    restore: RestoreApply | None = None
    roots: set[str] = field(default_factory=set)
    state: dict[str, Any] = field(default_factory=dict)


class WholeModelMetaTransform:
    """Coordinate a full meta copy and a list of FQN-scoped restructuring callbacks."""

    def __init__(self, source_model: Module, operations: Sequence[MetaOperation]):
        self.source_model = source_model
        self.model = copy_module_to_meta(source_model)
        self.operations = list(operations)
        self.transformed_fqns: set[str] = set()
        self.materialized_fqns: set[str] = set()
        self.restored_fqns: set[str] = set()
        self.sources_offloaded = False
        self.defer_subgraph_restore = False

        for operation in self.operations:
            operation.roots = operation.restructure_meta(
                self.model, self.transformed_fqns, operation.state
            )
            self.transformed_fqns.update(operation.roots)

    def offload_sources(self, onload_device: torch.device) -> None:
        """Offload the unchanged live source model before sequential calibration."""
        if self.sources_offloaded:
            return
        set_onload_device(self.source_model, onload_device)
        self.sources_offloaded = True

    def roots_for_subgraph(self, modules: Mapping[str, Module]) -> set[str]:
        return {
            fqn
            for fqn in self.transformed_fqns
            if _subgraph_owns_fqn(modules, fqn)
        }

    def materialize_subgraph(
        self,
        target_modules: Mapping[str, Module],
        target_offload_kwargs: dict[str, dict],
        device: torch.device,
    ) -> None:
        """Restructure loaded source modules and copy their state into meta targets."""
        roots = self.roots_for_subgraph(target_modules) - self.materialized_fqns
        source_modules = self._source_modules_for_subgraph(target_modules, roots)
        if not source_modules:
            return

        source_offload_kwargs = subgraph_onload_modules(source_modules)
        for operation in self.operations:
            operation_roots = operation.roots & roots
            if operation_roots:
                operation.restructure_source(
                    self.source_model, operation_roots, operation.state
                )

        populated_kwargs = {}
        for root in roots:
            source = _get_submodule(self.source_model, root)
            target = _get_submodule(self.model, root)
            cache_kwargs = source_offload_kwargs.get(root)
            if cache_kwargs is None:
                cache_kwargs = get_cache_init_kwargs(source)
            populated_kwargs.update(
                _qualify_kwargs(
                    root,
                    _copy_module_state(
                        target,
                        source,
                        device,
                        cache_kwargs,
                        recurse=True,
                    ),
                )
            )

        for fqn, target in target_modules.items():
            if _fqn_is_within(fqn, self.transformed_fqns):
                continue
            source = source_modules.get(fqn)
            if source is None or not _has_meta_parameters(target):
                continue
            cache_kwargs = source_offload_kwargs.get(fqn)
            if cache_kwargs is None:
                cache_kwargs = get_cache_init_kwargs(source)
            populated_kwargs.update(
                _qualify_kwargs(
                    fqn,
                    _copy_module_state(
                        target,
                        source,
                        device,
                        cache_kwargs,
                        recurse=False,
                    ),
                )
            )

        source_modules_after = self._source_modules_after_subgraph(source_modules, roots)
        subgraph_offload_modules(source_modules_after, source_offload_kwargs)
        self._offload_retained_sources(roots, source_offload_kwargs)

        target_offload_kwargs.update(populated_kwargs)
        target_offload_kwargs.update(subgraph_onload_modules(target_modules))
        self.materialized_fqns.update(roots)

    def materialize_all(self, device: torch.device) -> None:
        """Eagerly populate the full model for pipelines without subgraph support."""
        self.offload_sources(device)
        target_modules = dict(self.model.named_modules())
        offload_kwargs: dict[str, dict] = {}
        self.materialized_fqns.clear()
        self.materialize_subgraph(target_modules, offload_kwargs, device)
        subgraph_offload_modules(target_modules, offload_kwargs)
        self.defer_subgraph_restore = True

    def restore_subgraph(self, roots: set[str]) -> None:
        if self.defer_subgraph_restore:
            return
        self._restore_subgraph(roots)

    def _restore_subgraph(self, roots: set[str]) -> None:
        roots = roots - self.restored_fqns
        for root in roots - self.materialized_fqns:
            self.model.set_submodule(root, _get_submodule(self.source_model, root))
        for operation in self.operations:
            operation_roots = operation.roots & roots & self.materialized_fqns
            if operation.restore is not None and operation_roots:
                operation.restore(
                    self.model,
                    self.source_model,
                    operation_roots,
                    operation.state,
                )
        self.restored_fqns.update(roots)

    def restore_all(self) -> None:
        self.defer_subgraph_restore = False
        self._restore_subgraph(self.transformed_fqns - self.restored_fqns)

    def _source_modules_for_subgraph(
        self, target_modules: Mapping[str, Module], roots: set[str]
    ) -> dict[str, Module]:
        modules = {
            fqn: _get_submodule(self.source_model, fqn)
            for fqn, target in target_modules.items()
            if not _fqn_is_within(fqn, self.transformed_fqns)
            and _has_meta_parameters(target)
        }
        for root in roots:
            source = _get_submodule(self.source_model, root)
            modules.update(
                {
                    _qualify_fqn(root, relative_name): module
                    for relative_name, module in source.named_modules()
                }
            )
        return modules

    def _source_modules_after_subgraph(
        self, source_modules: Mapping[str, Module], roots: set[str]
    ) -> dict[str, Module]:
        modules = {
            fqn: module
            for fqn, module in source_modules.items()
            if not _fqn_is_within(fqn, self.transformed_fqns)
        }
        for root in roots:
            source = _get_submodule(self.source_model, root)
            modules.update(
                {
                    _qualify_fqn(root, relative_name): module
                    for relative_name, module in source.named_modules()
                }
            )
        return modules

    def _offload_retained_sources(
        self,
        roots: set[str],
        source_offload_kwargs: Mapping[str, dict],
    ) -> None:
        for operation in self.operations:
            if operation.restore is None:
                continue
            originals = operation.state.get("originals", {})
            saved_kwargs = operation.state.setdefault("source_offload_kwargs", {})
            for root in operation.roots & roots:
                original = originals.get(root)
                if original is None:
                    continue
                cache_kwargs = source_offload_kwargs.get(root)
                if cache_kwargs is None:
                    cache_kwargs = get_cache_init_kwargs(original)
                saved_kwargs[root] = cache_kwargs
                if not isinstance(original._parameters, OffloadCache):
                    offload_module(original, **cache_kwargs)


def build_meta_operations(
    *, layerwise_decompression: bool = False
) -> list[MetaOperation]:
    """Build restructuring callbacks applied to the meta model and loaded sources."""
    return [
        _moe_operation(layerwise_decompression),
        _offset_norm_operation(),
        _decompression_operation(layerwise_decompression),
    ]


def _moe_operation(decompress: bool) -> MetaOperation:
    def restructure_meta(
        model: Module, excluded_fqns: set[str], _state: dict[str, Any]
    ) -> set[str]:
        moes = get_non_linearized_moes(model)
        if moes and not isinstance(model, PreTrainedModel):
            raise TypeError("MoE linearization requires a PreTrainedModel root")
        roots = set()
        for fqn, source in moes:
            if _fqn_overlaps(fqn, excluded_fqns):
                continue
            meta_source = copy_module_to_meta(source)
            if decompress and _contains_quantized_module(source):
                _decompress_tree(meta_source)
            target = linearize_moe_module(meta_source, model, meta=True)
            model.set_submodule(fqn, target)
            roots.add(fqn)
        return roots

    def restructure_source(
        model: Module, roots: set[str], _state: dict[str, Any]
    ) -> None:
        for fqn in roots:
            source = _get_submodule(model, fqn)
            if decompress and _contains_quantized_module(source):
                _decompress_tree(source)
            target = linearize_moe_module(source, model)
            for module in target.modules():
                remove_module_offload(module, onload_tensors=True)
            model.set_submodule(fqn, target)

    return MetaOperation("linearize_moe", restructure_meta, restructure_source)


def _offset_norm_operation() -> MetaOperation:
    def restructure_meta(
        model: Module, excluded_fqns: set[str], state: dict[str, Any]
    ) -> set[str]:
        class_names = state.setdefault("class_names", {})
        roots = set()
        for fqn, source, class_name in get_norm_calibration_targets(model):
            if _fqn_overlaps(fqn, excluded_fqns):
                continue
            target = NormCalibrationModule.load_from_registry(
                class_name, original=source, config=model.config
            )
            model.set_submodule(fqn, target)
            class_names[fqn] = class_name
            roots.add(fqn)
        return roots

    def restructure_source(
        model: Module, roots: set[str], state: dict[str, Any]
    ) -> None:
        originals = state.setdefault("originals", {})
        class_names = state["class_names"]
        for fqn in roots:
            original = _get_submodule(model, fqn)
            target = NormCalibrationModule.load_from_registry(
                class_names[fqn], original=original, config=model.config
            )
            originals[fqn] = original
            model.set_submodule(fqn, target)

    def restore(
        model: Module,
        source_model: Module,
        roots: set[str],
        state: dict[str, Any],
    ) -> None:
        originals = state["originals"]
        saved_kwargs = state["source_offload_kwargs"]
        for fqn in roots:
            target = _get_submodule(model, fqn)
            original = originals.pop(fqn)
            remove_module_offload(target, onload_tensors=True)
            remove_module_offload(original, onload_tensors=True)
            if not isinstance(target, NormCalibrationModule):
                raise TypeError("Offset norm target must implement NormCalibrationModule")
            target.restore(original)
            offload_module(original, **saved_kwargs.pop(fqn))
            model.set_submodule(fqn, original)
            source_model.set_submodule(fqn, original)

    return MetaOperation(
        "offset_norm", restructure_meta, restructure_source, restore=restore
    )


def _decompression_operation(enabled: bool) -> MetaOperation:
    def restructure_meta(
        model: Module, excluded_fqns: set[str], _state: dict[str, Any]
    ) -> set[str]:
        if not enabled:
            return set()
        roots = set()
        for fqn, module in list(model.named_modules()):
            if not is_module_quantized(module) or _fqn_overlaps(fqn, excluded_fqns):
                continue
            decompress_module(module, leave_decompressed=False)
            roots.add(fqn)
        return roots

    def restructure_source(
        model: Module, roots: set[str], _state: dict[str, Any]
    ) -> None:
        for fqn in roots:
            module = _get_submodule(model, fqn)
            if is_module_quantized(module):
                decompress_module(module, leave_decompressed=False)

    return MetaOperation("decompress", restructure_meta, restructure_source)


def _copy_module_state(
    target: Module,
    source: Module,
    device: torch.device,
    offload_kwargs: dict,
    *,
    recurse: bool,
) -> dict[str, dict]:
    source_state = _named_state(source, recurse)
    target_state = _named_state(target, recurse)
    missing = set(source_state) - set(target_state)
    if missing:
        raise ValueError(f"Meta target is missing source state entries: {sorted(missing)}")
    defaults = {
        name
        for name, tensor in target_state.items()
        if tensor.device.type == "meta"
    }

    def initialize(module: Module) -> None:
        for name, tensor in _named_state(module, recurse).items():
            source_tensor = source_state.get(name)
            if source_tensor is None:
                if name in defaults:
                    tensor.fill_(1 if name.endswith("_scale") else 0)
                continue
            if source_tensor.device.type == "meta":
                raise ValueError(f"Source state {name} is still on the meta device")
            if tensor.shape != source_tensor.shape:
                raise ValueError(
                    f"Cannot copy {name}: target shape {tuple(tensor.shape)} does not "
                    f"match source shape {tuple(source_tensor.shape)}"
                )
            tensor.copy_(source_tensor.to(device=tensor.device, dtype=tensor.dtype))

    materialize_meta_module(target, device=device, initializer=initialize, recurse=recurse)
    modules = target.modules() if recurse else (target,)
    for module in modules:
        offload_module(module, **offload_kwargs)
    return {
        name: offload_kwargs for name, _ in target.named_modules()
    } if recurse else {"": offload_kwargs}


def _named_state(module: Module, recurse: bool) -> dict[str, torch.Tensor]:
    state = dict(module.named_parameters(remove_duplicate=False, recurse=recurse))
    state.update(dict(module.named_buffers(remove_duplicate=False, recurse=recurse)))
    return state


def _has_meta_parameters(module: Module) -> bool:
    return any(
        tensor is not None and tensor.device.type == "meta"
        for tensor in (*module._parameters.values(), *module._buffers.values())
    )


def _contains_quantized_module(module: Module) -> bool:
    return any(is_module_quantized(child) for child in module.modules())


def _decompress_tree(module: Module) -> None:
    for child in list(module.modules()):
        if is_module_quantized(child):
            decompress_module(child, leave_decompressed=False)


def _fqn_overlaps(fqn: str, roots: set[str]) -> bool:
    return any(
        fqn == root or fqn.startswith(f"{root}.") or root.startswith(f"{fqn}.")
        for root in roots
    )


def _fqn_is_within(fqn: str, roots: set[str]) -> bool:
    return any(not root or fqn == root or fqn.startswith(f"{root}.") for root in roots)


def _subgraph_owns_fqn(modules: Mapping[str, Module], fqn: str) -> bool:
    return fqn in modules or any(name.startswith(f"{fqn}.") for name in modules)


def _qualify_fqn(root: str, relative: str) -> str:
    return f"{root}.{relative}" if root and relative else root or relative


def _qualify_kwargs(root: str, relative_kwargs: Mapping[str, dict]) -> dict[str, dict]:
    return {_qualify_fqn(root, name): kwargs for name, kwargs in relative_kwargs.items()}


def _get_submodule(module: Module, fqn: str) -> Module:
    return module.get_submodule(fqn) if fqn else module
