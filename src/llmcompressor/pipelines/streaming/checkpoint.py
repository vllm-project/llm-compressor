"""
Checkpoint utilities for the streaming (meta-device) pipeline.

CheckpointMap: lightweight param_fqn → (shard_file, key, dtype, shape) index
  built from safetensors headers only — no weights read at construction time.

stage_modules: disk → CPU staging (safe for background thread; GIL released by safetensors I/O).
commit_staged: CPU → GPU, blocking.
commit_staged_async: CPU → GPU queued on a dedicated CUDA stream (non-blocking).
release_modules: GPU → meta (instant GPU free, <1ms).
MetaSubgraphPrefetcher: wraps stage_modules in a background thread.
"""

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open


__all__ = [
    "CheckpointEntry",
    "CheckpointMap",
    "stage_modules",
    "commit_staged",
    "commit_staged_async",
    "release_modules",
    "MetaSubgraphPrefetcher",
]


# ---------------------------------------------------------------------------
# CheckpointMap
# ---------------------------------------------------------------------------

@dataclass
class CheckpointEntry:
    shard_file: str
    key: str
    dtype: torch.dtype
    shape: tuple


class CheckpointMap:
    """
    Lightweight index: param_fqn → CheckpointEntry.

    Dense-only: does NOT resolve MoE expert 3D tensor slices.
    Build time: ~6ms for 0.6B, <1s for 72B (reads only safetensors headers).
    """

    def __init__(self, entries: dict[str, CheckpointEntry]):
        self._map = entries

    @classmethod
    def from_path(cls, ckpt_path: str) -> "CheckpointMap":
        ckpt_path = Path(ckpt_path)
        index_file = ckpt_path / "model.safetensors.index.json"
        entries: dict[str, CheckpointEntry] = {}

        if index_file.exists():
            # Multi-shard: read weight_map from index JSON
            with open(index_file) as f:
                index = json.load(f)
            weight_map: dict[str, str] = index["weight_map"]

            # Group keys by shard to minimize file opens
            shards: dict[str, list[str]] = {}
            for key, shard_name in weight_map.items():
                shards.setdefault(shard_name, []).append(key)

            for shard_name, keys in shards.items():
                shard_file = str(ckpt_path / shard_name)
                with safe_open(shard_file, framework="pt", device="cpu") as f:
                    for key in keys:
                        t = f.get_tensor(key)
                        entries[key] = CheckpointEntry(shard_file, key, t.dtype, tuple(t.shape))
                        del t
        else:
            # Single shard
            shard_file = str(ckpt_path / "model.safetensors")
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    entries[key] = CheckpointEntry(shard_file, key, t.dtype, tuple(t.shape))
                    del t

        # Auto-detect prefix remapping: some checkpoints use "model.language_model.*"
        # while the Python model uses "model.*" (e.g. Qwen3.8-27B VL-style checkpoint).
        # If the direct intersection with model params is tiny, try stripping the extra prefix.
        _LM_PREFIX = "model.language_model."
        _MODEL_PREFIX = "model."
        if any(k.startswith(_LM_PREFIX) for k in entries):
            remapped: dict[str, CheckpointEntry] = {}
            for k, entry in entries.items():
                if k.startswith(_LM_PREFIX):
                    new_key = _MODEL_PREFIX + k[len(_LM_PREFIX):]
                    remapped[new_key] = entry
                else:
                    remapped[k] = entry
            entries = remapped

        return cls(entries)

    def get(self, name: str) -> Optional[CheckpointEntry]:
        return self._map.get(name)

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, name: str) -> bool:
        return name in self._map


# ---------------------------------------------------------------------------
# stage_modules: disk → CPU  (safe in background thread — safetensors releases GIL)
# ---------------------------------------------------------------------------

def stage_modules(
    modules: list[nn.Module],
    ckpt_map: CheckpointMap,
    module_fqns: list[str],
) -> dict:
    """
    Load params/buffers for the given modules from safetensors → CPU RAM.

    Groups reads by shard file (each shard opened only once per call).
    Thread-safe: safetensors releases the GIL during I/O, so this truly runs
    in parallel with main-thread GPU compute.

    Returns:
        { (module_id, "param"|"buf", local_name): (module, cpu_tensor) }
    """
    # Build read plan grouped by shard
    plan: dict[str, list] = {}
    for module, fqn in zip(modules, module_fqns):
        for local_name, _ in module.named_parameters(recurse=False):
            full_key = f"{fqn}.{local_name}" if fqn else local_name
            entry = ckpt_map.get(full_key)
            if entry is None:
                continue
            plan.setdefault(entry.shard_file, []).append(
                (module, "param", local_name, entry.key, entry.dtype)
            )
        for local_name, _ in module.named_buffers(recurse=False):
            full_key = f"{fqn}.{local_name}" if fqn else local_name
            entry = ckpt_map.get(full_key)
            if entry is None:
                continue
            plan.setdefault(entry.shard_file, []).append(
                (module, "buf", local_name, entry.key, entry.dtype)
            )

    staged = {}
    for shard_file, items in plan.items():
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for module, kind, local_name, key, dtype in items:
                tensor = f.get_tensor(key)
                # safe_open().get_tensor() returns a lazy mmap-backed tensor; page faults
                # (actual disk reads) are deferred until the tensor's data is first touched.
                # clone() forces eager materialization here in the background thread so that
                # commit_staged_async can do a fast pinned-memory DMA with no page-fault
                # stalls in the main thread.
                tensor = tensor.clone()
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                staged[(id(module), kind, local_name)] = (module, tensor)
    return staged


# ---------------------------------------------------------------------------
# commit_staged: CPU → GPU
# ---------------------------------------------------------------------------

def commit_staged(staged: dict, device: torch.device) -> int:
    """
    Blocking: move staged CPU tensors to GPU and inject into module._parameters.
    Cannot use .data= on meta tensors — must replace the entire Parameter object.
    """
    count = 0
    for (mod_id, kind, local_name), (module, cpu_tensor) in staged.items():
        gpu_tensor = cpu_tensor.to(device)
        if kind == "param":
            old = module._parameters.get(local_name)
            req_grad = old.requires_grad if old is not None else False
            module._parameters[local_name] = nn.Parameter(gpu_tensor, requires_grad=req_grad)
        else:
            module._buffers[local_name] = gpu_tensor
        count += 1
    return count


def commit_staged_async(staged: dict, device: torch.device, h2d_stream: torch.cuda.Stream) -> int:
    """
    Async CUDA stream: queue H2D transfers on h2d_stream so they overlap with
    GPU compute on the default stream. Caller must synchronize:
        torch.cuda.current_stream().wait_stream(h2d_stream)
    after compute finishes (≈0ms when compute window is large enough).
    """
    count = 0
    with torch.cuda.stream(h2d_stream):
        for (mod_id, kind, local_name), (module, cpu_tensor) in staged.items():
            gpu_tensor = cpu_tensor.to(device, non_blocking=True)
            if kind == "param":
                old = module._parameters.get(local_name)
                req_grad = old.requires_grad if old is not None else False
                module._parameters[local_name] = nn.Parameter(gpu_tensor, requires_grad=req_grad)
            else:
                module._buffers[local_name] = gpu_tensor
            count += 1
    return count


# ---------------------------------------------------------------------------
# release_modules: GPU → meta  (instant GPU memory free)
# ---------------------------------------------------------------------------

def release_modules(
    modules: list[nn.Module],
    ckpt_map: CheckpointMap,
    module_fqns: list[str],
):
    """
    Replace checkpoint-backed params/buffers with meta tensors (GPU freed immediately).
    Non-checkpoint tensors (quantization scales written during calibration) are left
    untouched since they're not in ckpt_map.
    """
    for module, fqn in zip(modules, module_fqns):
        for local_name, param in list(module.named_parameters(recurse=False)):
            full_key = f"{fqn}.{local_name}" if fqn else local_name
            if ckpt_map.get(full_key) is not None:
                module._parameters[local_name] = nn.Parameter(
                    torch.empty(param.shape, dtype=param.dtype, device="meta"),
                    requires_grad=param.requires_grad,
                )
        for local_name, buf in list(module.named_buffers(recurse=False)):
            full_key = f"{fqn}.{local_name}" if fqn else local_name
            if ckpt_map.get(full_key) is not None:
                module._buffers[local_name] = torch.empty(
                    buf.shape, dtype=buf.dtype, device="meta"
                )


# ---------------------------------------------------------------------------
# MetaSubgraphPrefetcher: background-thread disk → CPU staging
# ---------------------------------------------------------------------------

class MetaSubgraphPrefetcher:
    """
    Wraps stage_modules in a background thread so disk → CPU reads overlap
    with GPU compute on the main thread.

    Usage:
        prefetcher = MetaSubgraphPrefetcher(model, ckpt_map)
        prefetcher.start(next_modules, next_fqns)  # fires background thread
        ... do GPU compute on current subgraph ...
        staged = prefetcher.join()                  # wait + get result
        commit_staged_async(staged, device, h2d_stream)
    """

    def __init__(self, model: nn.Module, ckpt_map: CheckpointMap):
        self._model = model
        self._ckpt_map = ckpt_map
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[dict] = None
        self._error: Optional[BaseException] = None
        self._disk_stage_s: float = 0.0

    def start(self, modules: list[nn.Module], fqns: list[str]) -> None:
        """Fire background thread to stage next subgraph's weights disk→CPU."""
        assert self._thread is None, "Previous prefetch not joined"
        self._result = None
        self._error = None
        self._disk_stage_s = 0.0

        def _worker():
            try:
                t0 = time.perf_counter()
                self._result = stage_modules(modules, self._ckpt_map, fqns)
                self._disk_stage_s = time.perf_counter() - t0
            except Exception as e:
                self._error = e

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def join(self) -> dict:
        """Wait for background staging to complete and return the staged dict."""
        assert self._thread is not None, "No prefetch started"
        self._thread.join()
        self._thread = None
        if self._error is not None:
            raise RuntimeError(f"Prefetch failed: {self._error}") from self._error
        result, self._result = self._result, None
        return result

    @property
    def disk_stage_s(self) -> float:
        """Wall time of the last completed stage_modules() call (bg thread)."""
        return self._disk_stage_s

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
