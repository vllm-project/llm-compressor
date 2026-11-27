# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage, materialize, and offload the modules owned by a calibration subgraph."""

from __future__ import annotations

import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
from compressed_tensors.offload import (
    AsyncWriter,
    get_cache_init_kwargs,
    offload_module,
)
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.cache.disk_utils import disk_load_context
from compressed_tensors.offload.module import (
    remove_module_offload,
    stage_module_offload,
)
from torch.nn import Module

from llmcompressor.modeling.module_restructure import transfer_module_state

if TYPE_CHECKING:
    from llmcompressor.pipelines.sequential.restructure import RestructureHandler

__all__ = [
    "subgraph_stage_modules",
    "materialize_modules",
    "subgraph_offload_modules",
]


def _source_modules(
    modules: dict[str, Module], restructure_handler: RestructureHandler
) -> dict[str, Module]:
    """Resolve direct sources, expanding only trees owned by a restructure."""
    sources = {}
    # Ordinary containers may be persistent while their children belong to later
    # subgraphs. Only a restructure (such as MoE linearization) consumes a full
    # source tree whose descendants differ from those in the live module dict.
    for name, module in modules.items():
        entry = restructure_handler.mapping.get(module)
        if entry is None or entry["materialized"]:
            sources[name] = module
        elif entry["restructures"]:
            sources.update(entry["source"].named_modules(prefix=name))
        else:
            sources[name] = entry["source"]
    return sources


def subgraph_stage_modules(
    modules: dict[str, Module],
    restructure_handler: RestructureHandler,
    pin_memory: bool = False,
    async_writer: AsyncWriter | None = None,
    distributed_group=None,
) -> None:
    """Stage retained sources or previously materialized modules in CPU memory."""
    if distributed_group is not None:
        if async_writer is None:
            raise ValueError("Distributed staging requires an async disk writer")
        async_writer.flush_distributed(distributed_group)
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
    async_writer: AsyncWriter | None = None,
    distributed_group=None,
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
        async_writer=async_writer,
        distributed_group=distributed_group,
    )
    return modules, future


def materialize_modules(
    modules: dict[str, Module],
    restructure_handler: RestructureHandler,
    device: torch.device,
) -> dict[str, dict]:
    """Onload sources, apply ordered conversions, and fill initialized live modules.

    ``modules`` includes the live descendants owned by the subgraph, as returned by
    ``Subgraph.submodule_dict``. Retained source trees can have a different structure.
    """
    # get metadata for non materialized modules on current subgraph
    pending = {
        module: restructure_handler.mapping[module]
        for module in modules.values()
        if module in restructure_handler.mapping
        and not restructure_handler.mapping[module]["materialized"]
    }

    # 1. Onload the source modules
    sources = _source_modules(modules, restructure_handler)
    offload_kwargs = {}
    with disk_load_context():
        for name, source in sources.items():
            if isinstance(source._parameters, OffloadCache):
                offload_kwargs[name] = get_cache_init_kwargs(source)
                remove_module_offload(source, onload_tensors=True)

    # 2. Restructure the source modules (same as what we did with
    # target modules before init). These ones are real weights.
    for restructure in restructure_handler.restructures:
        selected = {
            target: entry
            for target, entry in pending.items()
            if restructure in entry["restructures"]
        }
        restructure.restructure_sources(selected, device, offload_kwargs)

    # 3. Move the weights from the restructured source to the
    # target module on the model. Delete metadata if no restore needed.
    for target, entry in pending.items():
        # Replacements were transferred by the restructure that owns them.
        if not entry["materialized"]:
            transfer_module_state(target, entry["source"], device)
            entry["materialized"] = True
        if not any(
            restructure.requires_restore for restructure in entry["restructures"]
        ):
            del restructure_handler.mapping[target]
    return offload_kwargs


def subgraph_offload_modules(
    modules: dict[str, Module],
    offload_kwargs: dict[str, dict],
    default: dict | None = None,
    async_writer: AsyncWriter | None = None,
) -> None:
    """Offload exactly the supplied modules using saved or inherited settings.

    Modules without settings for themselves or an ancestor retain the device-based
    fallback, overridden by ``default`` when provided.
    """

    def get_offload_kwargs(name: str, module: Module) -> dict:
        offload_name = name
        while offload_name not in offload_kwargs and offload_name:
            offload_name = offload_name.rpartition(".")[0]

        if offload_name in offload_kwargs:
            module_kwargs = offload_kwargs[offload_name].copy()
        else:
            module_kwargs = get_cache_init_kwargs(module)
            if default is not None:
                module_kwargs.update(default)
        if async_writer is not None and async_writer.streaming_save:
            module_kwargs["offload_device"] = "disk"
            if module_kwargs.get("offload_dir") is None:
                output_dir = async_writer.output_dir
                if output_dir is None:
                    raise ValueError("Streaming save requires an output directory")
                cache_dir = output_dir.parent / f".{output_dir.name}-offload"
                cache_dir.mkdir(parents=True, exist_ok=True)
                module_kwargs["offload_dir"] = str(cache_dir)
            module_kwargs["module_name"] = name
        if (
            async_writer is not None
            and str(module_kwargs.get("offload_device")) == "disk"
        ):
            module_kwargs["async_writer"] = async_writer
        return module_kwargs

    with contextlib.ExitStack() as stack:
        if async_writer is not None:
            stack.enter_context(async_writer.batch())
        for name, module in modules.items():
            offload_module(module, **get_offload_kwargs(name, module))
