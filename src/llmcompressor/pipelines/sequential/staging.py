"""Stage disk-backed subgraphs with direct reads into CPU tensor storage."""

from __future__ import annotations

import os
import sys

import torch
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.utils import catch_pinned_mem_error
from compressed_tensors.offload.module import (
    subgraph_stage_modules as upstream_stage_modules,
)
from compressed_tensors.offload.utils import to_tensor
from loguru import logger

from llmcompressor.pipelines.sequential._staging_io import read_tensors


@catch_pinned_mem_error
def subgraph_stage_modules(
    modules: dict[str, torch.nn.Module], *, pin_memory: bool = False
) -> None:
    """Read a subgraph's disk tensors into fresh CPU allocations before onload.

    Allocate pinned storage directly when requested for accelerator execution.
    Preserve the upstream staging behavior for unsupported platforms, cache
    implementations and tensor subclasses. Python still manages requests and
    tensor metadata; this does not release the GIL across the entire batch.
    """
    if sys.byteorder != "little" or not hasattr(os, "preadv"):
        upstream_stage_modules(modules, pin_memory=pin_memory)
        return

    requests, destinations, pending, fallback = [], [], [], {}
    can_pin = pin_memory and torch.accelerator.is_available()
    for name, module in modules.items():
        caches = (module._parameters, module._buffers)
        # Custom/distributed caches and tensor subclasses have their own semantics.
        if not all(
            type(cache) is DiskCache and not cache.is_staged for cache in caches
        ):
            fallback[name] = module
            continue
        if any(
            tensor is not None
            and (
                type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                or tensor.layout != torch.strided
                or tensor.is_quantized
            )
            for cache in caches
            for tensor in cache.offloaded_values.values()
        ):
            fallback[name] = module
            continue

        mappings = []
        for cache in caches:
            staged = dict.fromkeys(cache.offloaded_values)
            mappings.append(staged)
            pinned = can_pin and torch.device(cache.onload_device).type != "cpu"
            for key, template in cache.offloaded_values.items():
                if template is None:
                    continue
                info = cache.index[template]
                requests.append(
                    (
                        os.fspath(info["safetensors_file"]),
                        info["weight_name"],
                        getattr(torch, info["dtype"]),
                        pinned,
                    )
                )
                destinations.append((staged, key, template, cache))
        pending.append((module, caches, mappings))

    if requests:
        logger.bind(log_once=True).info(
            "Using direct disk reads for sequential staging"
        )
        tensors = read_tensors(requests)
        for tensor, (mapping, key, template, cache) in zip(tensors, destinations):
            # Let the upstream reader handle future safetensors dtypes.
            mapping[key] = (
                cache.stage(template, pin_memory=pin_memory)
                if tensor is None
                else to_tensor(tensor, template)
            )

    # Keep existing cache mappings intact until reads and metadata restoration succeed.
    for module, caches, mappings in pending:
        for cache, mapping in zip(caches, mappings):
            cache.offloaded_values = mapping
            cache.is_staged = True
        module.forward = module._original_forward_func.__get__(module)
        del module._original_forward_func
    upstream_stage_modules(fallback, pin_memory=pin_memory)
