"""Batch disk staging with a native worker that releases the Python GIL."""

from __future__ import annotations

import os
import sys

import torch
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.utils import catch_pinned_mem_error
from compressed_tensors.offload.module import (
    subgraph_stage_modules as python_stage_modules,
)
from compressed_tensors.offload.utils import to_tensor
from loguru import logger

try:
    from llmcompressor import _staging_native
except ImportError:
    _staging_native = None


@catch_pinned_mem_error
def subgraph_stage_modules(
    modules: dict[str, torch.nn.Module], pin_memory: bool = False
) -> None:
    """Stage disk caches without holding the GIL during native allocation/I/O.

    Python prepares requests and installs the completed tensors. The native call
    owns fresh tensors and releases the GIL across file opening, header parsing,
    allocation, reading and dtype conversion for the whole batch. No module or
    Python object is accessed there. CPU/custom/distributed caches retain their
    own staging implementation. The executor's Future remains the synchronization
    point before the main thread consumes staged modules.
    """
    if (
        _staging_native is None
        or sys.byteorder != "little"
        or os.environ.get("LLMCOMPRESSOR_NATIVE_STAGING", "1") == "0"
    ):
        python_stage_modules(modules, pin_memory=pin_memory)
        return

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
        logger.bind(log_once=True).info("Using native disk staging with GIL release")
        tensors = _staging_native.read_batch(requests)
        for tensor, (mapping, key, template, cache) in zip(tensors, destinations):
            # Future safetensors dtypes can continue to use the upstream reader.
            mapping[key] = (
                cache.stage(template, pin_memory=pin_memory)
                if tensor is None
                else to_tensor(tensor, template)
            )

    # Publish only after every native read and metadata conversion succeeded.
    for module, caches, mappings in pending:
        for cache, mapping in zip(caches, mappings):
            cache.offloaded_values = mapping
            cache.is_staged = True
        module.forward = module._original_forward_func.__get__(module)
        del module._original_forward_func
    python_stage_modules(fallback, pin_memory=pin_memory)
