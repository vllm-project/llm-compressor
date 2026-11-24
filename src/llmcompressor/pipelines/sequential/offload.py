# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage, materialize, and offload the modules owned by a calibration subgraph."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.cache.disk_utils import disk_load_context
from compressed_tensors.offload.module import (
    remove_module_offload,
    stage_module_offload,
)
from torch.nn import Module

from llmcompressor.modeling.module_restructure import (
    clear_module_state,
    copy_module_state,
)

if TYPE_CHECKING:
    from llmcompressor.pipelines.sequential.restructure import RestructureHandler

__all__ = [
    "subgraph_stage_modules",
    "subgraph_onload_modules",
    "subgraph_offload_modules",
]


def _source_modules(
    modules: dict[str, Module], restructure_handler: RestructureHandler
) -> dict[str, Module]:
    """Resolve delayed targets to their real source trees; keep live modules direct."""
    sources = {}
    for name, module in modules.items():
        source = restructure_handler.mapping.get(module)
        if source is None:
            sources[name] = module
        else:
            sources.update(source.named_modules(prefix=name))
    return sources


def subgraph_stage_modules(
    modules: dict[str, Module],
    restructure_handler: RestructureHandler,
    pin_memory: bool = False,
) -> None:
    """Stage retained sources or previously materialized modules in CPU memory."""
    with disk_load_context():
        for module in _source_modules(modules, restructure_handler).values():
            cache = module._parameters
            if isinstance(cache, OffloadCache) and not cache.is_staged:
                stage_module_offload(module, pin_memory=pin_memory)


def _submit_subgraph_staging(
    executor: ThreadPoolExecutor,
    modules: dict[str, Module],
    current_modules: dict[str, Module],
    restructure_handler: RestructureHandler,
    pin_memory: bool,
) -> tuple[dict[str, Module], Future] | None:
    """Prefetch only when both live modules and retained sources are disjoint."""
    if set(modules.values()) & set(current_modules.values()):
        return None
    sources = _source_modules(modules, restructure_handler)
    current_sources = _source_modules(current_modules, restructure_handler)
    if set(sources.values()) & set(current_sources.values()):
        return None

    future = executor.submit(
        subgraph_stage_modules,
        modules,
        restructure_handler,
        pin_memory=pin_memory,
    )
    return modules, future


def subgraph_onload_modules(
    modules: dict[str, Module],
    restructure_handler: RestructureHandler,
    device: torch.device,
) -> dict[str, dict]:
    """Onload sources, apply ordered conversions, and fill initialized live modules.

    ``modules`` includes the live descendants owned by the subgraph, as returned by
    ``Subgraph.submodule_dict``. Retained source trees can have a different structure.
    """
    pending = {
        name: module
        for name, module in modules.items()
        if module in restructure_handler.mapping
    }
    sources = _source_modules(modules, restructure_handler)
    offload_kwargs = {}
    with disk_load_context():
        for name, source in sources.items():
            if isinstance(source._parameters, OffloadCache):
                offload_kwargs[name] = get_cache_init_kwargs(source)
                remove_module_offload(source, onload_tensors=True)

    for restructure in restructure_handler.restructures:
        restructure.materialize(
            restructure_handler.model, sources, device, offload_kwargs
        )
    for target in pending.values():
        # Replacements were copied by the restructure that owns them.
        source = restructure_handler.mapping.pop(target, None)
        if source is not None:
            copy_module_state(target, source, device, recurse=False)
            clear_module_state(source)
    return offload_kwargs


def subgraph_offload_modules(
    modules: dict[str, Module],
    offload_kwargs: dict[str, dict],
    default: dict | None = None,
) -> None:
    """Offload scoped modules and new descendants using saved or inherited settings.

    Modules without settings for themselves or an ancestor retain the device-based
    fallback, overridden by ``default`` when provided. Already offloaded descendants
    outside the supplied scope, including staged modules, are left untouched.
    """
    scoped_modules = set(modules.values())

    def get_offload_kwargs(name: str, module: Module) -> dict:
        offload_name = name
        while offload_name not in offload_kwargs and offload_name:
            offload_name = offload_name.rpartition(".")[0]

        if offload_name in offload_kwargs:
            return offload_kwargs[offload_name]
        module_kwargs = get_cache_init_kwargs(module)
        if default is not None:
            module_kwargs.update(default)
        return module_kwargs

    def offload_new_descendants(name: str, module: Module) -> None:
        for child_name, child in module.named_children():
            if child in scoped_modules or isinstance(child._parameters, OffloadCache):
                continue

            child_name = f"{name}.{child_name}" if name else child_name
            offload_module(child, **get_offload_kwargs(child_name, child))
            offload_new_descendants(child_name, child)

    for name, module in modules.items():
        offload_module(module, **get_offload_kwargs(name, module))

    for name, module in modules.items():
        offload_new_descendants(name, module)
