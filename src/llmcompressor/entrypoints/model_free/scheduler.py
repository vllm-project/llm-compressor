"""
Memory-aware job scheduler for multi-GPU model-free PTQ.

The old round-robin assignment bakes device choices into jobs at build time,
which blows up when shards differ in size or GPUs differ in speed — the slow
GPU accumulates pending work and eventually OOMs.

This module replaces that with a capacity-first scheduler: before submitting
a job the main thread checks ``torch.get_device_module(...).mem_get_info``
and picks the GPU with the most headroom.  If nothing fits, it waits for a
running job to finish and retries.
"""

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Optional

import torch
import tqdm
from compressed_tensors.utils.safetensors_load import InverseWeightMap
from loguru import logger

__all__ = ["estimate_job_memory", "exec_jobs_dynamic"]

# 3x covers the weight tensors themselves plus quantization intermediates
# (scales, zero-points, compressed output buffers).  Conservative on purpose
# — a slight overestimate just means we serialize a bit more; an underestimate
# means OOM.
_MEMORY_MULTIPLIER = 3.0


def estimate_job_memory(inverse_weight_map: InverseWeightMap) -> int:
    """Rough memory estimate for a shard job, based on source file sizes."""
    total = sum(os.path.getsize(p) for p in inverse_weight_map)
    return int(total * _MEMORY_MULTIPLIER)


def _free_bytes(dev: torch.device, reserved: dict[torch.device, int]) -> int | float:
    """Driver-reported free VRAM minus bytes already promised to queued
    jobs.  Returns ``inf`` for CPU devices (unbounded capacity)."""
    if dev.type == "cpu":
        return float("inf")
    module = torch.get_device_module(dev)
    mem_get_info = getattr(module, "mem_get_info", None)
    if mem_get_info is None:
        raise RuntimeError(f"{dev.type!r} backend does not support mem_get_info")
    free, _ = mem_get_info(dev)
    return max(0, free - reserved.get(dev, 0))


def _empty_cache(dev: torch.device) -> None:
    """Flush the caching allocator if the backend supports it."""
    if dev.type == "cpu":
        return
    module = torch.get_device_module(dev)
    empty_cache = getattr(module, "empty_cache", None)
    if empty_cache is not None:
        empty_cache()


def _pick_device(
    devices: list[torch.device],
    required: int,
    reserved: dict[torch.device, int],
) -> Optional[torch.device]:
    """Return the device with the most free VRAM that can fit *required*
    bytes, or ``None`` if nothing qualifies right now."""
    best, best_free = None, -1
    for dev in devices:
        if dev.type == "cpu":
            return dev
        available = _free_bytes(dev, reserved)
        if available >= required and available > best_free:
            best, best_free = dev, available
    return best


def exec_jobs_dynamic(
    jobs: list[tuple],
    devices: list[torch.device],
    max_workers: int,
    memory_estimates: list[int],
    desc: str = "Quantizing",
) -> list:
    """Run *jobs* across *devices*, assigning each job at submit time to
    whichever GPU has the most free memory.

    Each job tuple is ``(fn, inverse_weight_map, save_path, scheme, ignore,
    converter)`` — same as ``_build_jobs`` output, deliberately **without** a
    device field.  The device gets spliced in right before ``executor.submit``.

    Effective concurrency is capped by measured GPU capacity: even if
    ``max_workers`` is high, jobs are held back until a GPU can actually fit
    the estimated footprint.
    """
    n = len(jobs)

    # CPU path — nothing to schedule
    if all(d.type == "cpu" for d in devices):
        out = []
        for job in tqdm.tqdm(jobs, desc=desc):
            fn, iwm, sp, sch, ign, conv = job
            out.append(fn(iwm, sp, sch, ign, devices[0], conv))
        return out

    reserved = {d: 0 for d in devices}

    # Single worker — simple loop, useful for debugging
    if max_workers == 1:
        out = []
        for i, job in enumerate(tqdm.tqdm(jobs, desc=desc)):
            fn, iwm, sp, sch, ign, conv = job
            dev = _pick_device(devices, memory_estimates[i], reserved)
            if dev is None:
                dev = max(
                    devices,
                    key=lambda d: _free_bytes(d, reserved),
                )
                logger.warning(
                    f"Shard {i} exceeds estimated capacity on all "
                    f"devices, falling back to {dev}"
                )
            logger.debug(
                f"Shard {i} -> {dev} " f"(~{memory_estimates[i] / 1e9:.2f} GB)"
            )
            res = fn(iwm, sp, sch, ign, dev, conv)
            out.append(res)
            _empty_cache(dev)
        return out

    # Multi-worker: main thread schedules, workers execute
    results = [None] * n
    pending = list(range(n))
    fut_device: dict = {}  # future -> device, for releasing reservations

    with tqdm.tqdm(total=n, desc=desc) as bar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            inflight: dict = {}  # future -> job index

            while pending or inflight:
                # --- try to fill idle workers ----------------------------
                submitted = []
                for idx in pending:
                    if len(inflight) >= max_workers:
                        break
                    dev = _pick_device(devices, memory_estimates[idx], reserved)
                    if dev is None:
                        continue

                    fn, iwm, sp, sch, ign, conv = jobs[idx]
                    fut = pool.submit(fn, iwm, sp, sch, ign, dev, conv)
                    inflight[fut] = idx
                    fut_device[fut] = dev
                    reserved[dev] += memory_estimates[idx]
                    submitted.append(idx)
                    logger.debug(
                        f"Shard {idx} -> {dev} "
                        f"(~{memory_estimates[idx] / 1e9:.2f} GB)"
                    )

                for idx in submitted:
                    pending.remove(idx)

                # --- nothing running and nothing fits: force it ----------
                if not inflight:
                    if not pending:
                        break
                    # pick smallest job, most-free device — better than
                    # silently dropping work
                    idx = min(pending, key=lambda i: memory_estimates[i])
                    dev = max(
                        devices,
                        key=lambda d: _free_bytes(d, reserved),
                    )
                    logger.warning(
                        f"Shard {idx} "
                        f"(~{memory_estimates[idx] / 1e9:.2f} GB) "
                        f"exceeds estimated capacity on all devices; "
                        f"forcing onto {dev}"
                    )
                    fn, iwm, sp, sch, ign, conv = jobs[idx]
                    fut = pool.submit(fn, iwm, sp, sch, ign, dev, conv)
                    inflight[fut] = idx
                    fut_device[fut] = dev
                    reserved[dev] += memory_estimates[idx]
                    pending.remove(idx)
                    continue

                # --- wait for at least one job to finish -----------------
                done, _ = wait(
                    inflight.keys(),
                    timeout=2.0,
                    return_when=FIRST_COMPLETED,
                )

                for f in done:
                    i = inflight.pop(f)
                    dev = fut_device.pop(f)
                    reserved[dev] -= memory_estimates[i]
                    results[i] = f.result()  # propagates exceptions
                    bar.update(1)
                    # flush the caching allocator so mem_get_info
                    # reflects actually-free memory next pass
                    _empty_cache(dev)

    return results
