"""Prefetch disk tensors in a separate Python process with its own GIL."""

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

from llmcompressor.pipelines.sequential.staging_process import ProcessReader


def _supports_process_staging() -> bool:
    return (
        sys.platform.startswith("linux")
        and sys.byteorder == "little"
        and hasattr(os, "preadv")
        and hasattr(os, "posix_fallocate")
    )


class ModuleStager:
    """One pipeline's persistent disk reader, used by its staging thread.

    Enter before creating the staging executor, then join the executor before
    closing this context. The worker handles parsing and file reads; the parent
    retains ownership of module caches, tensor wrapping, pinning and CUDA.
    """

    def __init__(self):
        self.reader: ProcessReader | None = None
        self.entered = False
        self.closed = False

    def __enter__(self):
        if self.entered or self.closed:
            raise RuntimeError("Module stager contexts cannot be reused")
        self.entered = True
        if _supports_process_staging():
            self.reader = ProcessReader()
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self.closed = True
        if self.reader is not None:
            self.reader.close()

    @catch_pinned_mem_error
    def stage(self, modules: dict[str, torch.nn.Module], pin_memory: bool = False):
        if self.closed or not self.entered:
            raise RuntimeError("Module stager must be used inside its context")
        if self.reader is None:
            upstream_stage_modules(modules, pin_memory=pin_memory)
            return
        _stage_modules(modules, pin_memory, self.reader.read_batch)


def subgraph_stage_modules(modules, pin_memory=False):
    """Stage a single batch; pipelines retain a ModuleStager across subgraphs."""
    with ModuleStager() as stager:
        stager.stage(modules, pin_memory=pin_memory)


def _stage_modules(modules, pin_memory, reader):
    # Prepare plain read requests while holding the GIL. Only the parent process
    # touches modules or restores metadata; publish after the whole batch succeeds.
    requests, destinations, pending, fallback = [], [], [], {}
    can_pin = pin_memory and torch.accelerator.is_available()
    for name, module in modules.items():
        caches = (module._parameters, module._buffers)
        # Exact types: custom/distributed caches can have additional semantics.
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
        logger.bind(log_once=True).info("Using a Python process for disk staging")
        tensors = reader(requests)
        for tensor, (mapping, key, template, cache) in zip(tensors, destinations):
            # Future safetensors dtypes can continue to use the upstream reader.
            mapping[key] = (
                cache.stage(template, pin_memory=pin_memory)
                if tensor is None
                else to_tensor(tensor, template)
            )

    # Publish only after every read and metadata conversion succeeded.
    for module, caches, mappings in pending:
        for cache, mapping in zip(caches, mappings):
            cache.offloaded_values = mapping
            cache.is_staged = True
        module.forward = module._original_forward_func.__get__(module)
        del module._original_forward_func
    upstream_stage_modules(fallback, pin_memory=pin_memory)
