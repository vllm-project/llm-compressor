# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in host-side attribution at existing library calls; no added device waits."""

import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from importlib import import_module

import torch
from compressed_tensors import ModelCompressor
from compressed_tensors.offload.cache import (
    CPUCache,
    DiskCache,
    DistributedCPUCache,
    DistributedDiskCache,
)
from compressed_tensors.utils import patch_attr

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.quarot import base as quarot
from llmcompressor.transformers.compression import compressed_tensors_utils as saving


def _counter():
    return {"calls": 0, "host_seconds": 0.0, "logical_bytes": 0, "failures": 0}


class PhaseAttribution:
    def __init__(self, report):
        self.phase = ContextVar("glm_profile_phase", default="outside")
        self.state_depth = ContextVar("glm_profile_state_depth", default=0)
        self.operations = defaultdict(_counter)
        self.caches = defaultdict(_counter)
        self.segments = []
        self.progress_state = {}
        self.details = {
            "timing_kind": "inclusive_host_wall_no_added_sync",
            "limitations": (
                "Nested calls overlap; logical bytes are not physical I/O. "
                "Async collective timings stop at submission. Python sync hooks "
                "cannot see all internal device waits. Segment time is host wall."
            ),
            "operations": self.operations,
            "cache_by_phase_backend": self.caches,
            "quarot_segments": self.segments,
            "unavailable_hooks": [],
        }
        report["attribution"] = self.details

    @contextmanager
    def measure(self, name, *, phase=None, counters=None, logical_bytes=0):
        token = self.phase.set(phase) if phase is not None else None
        key = f"{self.phase.get()}/{name}"
        entry = (self.operations if counters is None else counters)[key]
        start = time.perf_counter()
        try:
            yield
        except BaseException:
            entry["failures"] += 1
            raise
        finally:
            entry["calls"] += 1
            entry["host_seconds"] += time.perf_counter() - start
            entry["logical_bytes"] += logical_bytes
            if token is not None:
                self.phase.reset(token)

    def wrap(self, original, name, *, phase=None):
        @wraps(original)
        def call(*args, **kwargs):
            with self.measure(name, phase=phase):
                return original(*args, **kwargs)

        return call

    def cache_call(self, original, label):
        @wraps(original)
        def call(cache, value, *args, **kwargs):
            size = value.numel() * value.element_size() if value is not None else 0
            with self.measure(label, counters=self.caches, logical_bytes=size):
                return original(cache, value, *args, **kwargs)

        return call

    def progress(self, original):
        def track(progress, items):
            phase = "quarot." + progress.phase.replace(" ", "_")
            token = self.phase.set(phase)
            state = self.progress_state.setdefault(
                progress, {"completed": 0, "time": time.perf_counter(), "caches": {}}
            )
            try:
                for item in original(progress, items):
                    yield item
                    # The original tracker increments when its yield resumes next.
                    completed = progress.completed + 1
                    if (
                        completed % progress.interval == 0
                        or completed == progress.total
                    ):
                        current = {
                            k: dict(v)
                            for k, v in self.caches.items()
                            if k.startswith(phase + "/")
                        }
                        now = time.perf_counter()
                        self.segments.append(
                            {
                                "phase": phase,
                                "from_completed": state["completed"],
                                "to_completed": completed,
                                "total": progress.total,
                                "host_seconds": now - state["time"],
                                "cache_delta": {
                                    k: {
                                        field: v[field]
                                        - state["caches"].get(k, {}).get(field, 0)
                                        for field in v
                                    }
                                    for k, v in current.items()
                                },
                            }
                        )
                        state.update(completed=completed, time=now, caches=current)
            finally:
                self.phase.reset(token)

        return track

    @contextmanager
    def instrument(self, root):
        init = import_module("compressed_tensors.quantization.lifecycle.initialize")
        apply = import_module("compressed_tensors.quantization.lifecycle.apply")
        hf = import_module("transformers.modeling_utils")
        with ExitStack() as stack:

            def bind(owner, name, label, phase=None):
                if not hasattr(owner, name):
                    self.details["unavailable_hooks"].append(label)
                    return
                stack.enter_context(
                    patch_attr(
                        owner, name, self.wrap(getattr(owner, name), label, phase=phase)
                    )
                )

            bind(QuantizationModifier, "on_initialize", "total", "initialization")
            bind(apply, "initialize_module_for_quantization", "initialize_module")
            bind(init, "initialize_qparams", "initialize_qparams")
            # Local constructor binding only; torch.nn.Parameter remains a class.
            bind(init, "Parameter", "qparam_parameter_construct")
            bind(torch.nn.Module, "register_parameter", "register_parameter")
            bind(quarot.QuaRotModifier, "on_calibration_start", "total", "quarot")
            stack.enter_context(
                patch_attr(
                    quarot._QuaRotProgress,
                    "track",
                    self.progress(quarot._QuaRotProgress.track),
                )
            )
            for cache in (
                CPUCache,
                DiskCache,
                DistributedCPUCache,
                DistributedDiskCache,
            ):
                for name in ("onload", "offload", "update_offload"):
                    # Patch each implementation once. Distributed offload includes
                    # the base transfer and shared-backing collectives: nested totals.
                    if name not in cache.__dict__:
                        continue
                    stack.enter_context(
                        patch_attr(
                            cache,
                            name,
                            self.cache_call(
                                getattr(cache, name), f"{cache.__name__}.{name}"
                            ),
                        )
                    )
            bind(ModelCompressor, "compress_model", "total", "save.compress")
            bind(saving, "to_accelerate", "total", "save.to_accelerate")
            bind(saving, "from_accelerate", "total", "save.restore_dispatch")
            bind(hf, "split_torch_state_dict_into_shards", "reshard")
            bind(hf, "safe_save_file", "write_shard")
            original_state_dict = torch.nn.Module.state_dict

            def state_dict(module, *args, **kwargs):
                if self.state_depth.get() or not self.phase.get().startswith("save"):
                    return original_state_dict(module, *args, **kwargs)
                token = self.state_depth.set(1)
                try:
                    with self.measure("state_dict_root"):
                        return original_state_dict(module, *args, **kwargs)
                finally:
                    self.state_depth.reset(token)

            stack.enter_context(patch_attr(torch.nn.Module, "state_dict", state_dict))
            for name in (
                "barrier",
                "all_reduce",
                "all_gather",
                "broadcast",
                "all_gather_object",
                "broadcast_object_list",
            ):
                bind(torch.distributed, name, "collective." + name)
            if hasattr(torch, "npu"):
                bind(torch.npu, "synchronize", "explicit_npu_synchronize")
            with self.measure("total", phase=root):
                yield
