"""
Memory-aware job scheduler for multi-GPU model-free PTQ.

The old round-robin assignment bakes device choices into jobs at build time,
which blows up when shards differ in size or GPUs differ in speed — the slow
GPU accumulates pending work and eventually OOMs.

This module replaces that with a capacity-first scheduler: free VRAM is
queried once via ``torch.accelerator.memory.get_memory_info`` and then
tracked locally through reservation accounting.  The GPU with the most
headroom gets the next job; if nothing fits, we wait for a running job to
finish and retry.
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


def _snapshot_free(devices: list[torch.device]) -> dict[torch.device, int]:
    """Query free VRAM once per device.  CPU devices are skipped."""
    free = {}
    for d in devices:
        if d.type != "cpu":
            mem_free, _ = torch.accelerator.memory.get_memory_info(d)
            free[d] = mem_free
    return free


def _free_bytes(
    dev: torch.device,
    initial_free: dict[torch.device, int],
    reserved: dict[torch.device, int],
) -> int | float:
    """Available VRAM for *dev*: initial snapshot minus promised reservations.
    Returns ``inf`` for CPU devices (unbounded capacity)."""
    if dev.type == "cpu":
        return float("inf")
    return max(0, initial_free.get(dev, 0) - reserved.get(dev, 0))


def _pick_device(
    devices: list[torch.device],
    required: int,
    initial_free: dict[torch.device, int],
    reserved: dict[torch.device, int],
) -> Optional[torch.device]:
    """Return the device with the most available VRAM that can fit *required*
    bytes, or ``None`` if nothing qualifies."""
    best, best_free = None, -1
    for dev in devices:
        if dev.type == "cpu":
            return dev
        available = _free_bytes(dev, initial_free, reserved)
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

    Free VRAM is queried once at startup; subsequent scheduling decisions
    rely on reservation accounting so we never re-query the driver in a hot
    loop.  Effective concurrency is capped by estimated GPU capacity: even
    if ``max_workers`` is high, jobs are held back until a GPU can actually
    fit the estimated footprint.
    """
    n = len(jobs)

    # CPU path — nothing to schedule
    if all(d.type == "cpu" for d in devices):
        out = []
        for job in tqdm.tqdm(jobs, desc=desc):
            fn, iwm, sp, sch, ign, conv = job
            out.append(fn(iwm, sp, sch, ign, devices[0], conv))
        return out

    # Snapshot free VRAM once; all later decisions use accounting only
    initial_free = _snapshot_free(devices)

    # Single worker — pick the best device once upfront
    if max_workers == 1:
        device = max(initial_free, key=initial_free.get)
        out = []
        for i, job in enumerate(tqdm.tqdm(jobs, desc=desc)):
            if memory_estimates[i] > initial_free[device]:
                logger.warning(
                    f"Shard {i} (~{memory_estimates[i] / 1e9:.2f} GB) "
                    f"exceeds estimated capacity of {device}"
                )
            fn, iwm, sp, sch, ign, conv = job
            out.append(fn(iwm, sp, sch, ign, device, conv))
        return out

    # Multi-worker: main thread schedules, workers execute
    reserved = {d: 0 for d in devices}
    results = [None] * n
    pending = list(range(n))
    fut_device: dict = {}  # future -> device, for releasing reservations

    with tqdm.tqdm(total=n, desc=desc) as bar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            inflight: dict = {}  # future -> job index

            while pending or inflight:
                # --- try to fill idle workers ---
                for idx in list(pending):
                    if len(inflight) >= max_workers:
                        break
                    dev = _pick_device(
                        devices,
                        memory_estimates[idx],
                        initial_free,
                        reserved,
                    )
                    if dev is None:
                        continue

                    fn, iwm, sp, sch, ign, conv = jobs[idx]
                    fut = pool.submit(fn, iwm, sp, sch, ign, dev, conv)
                    inflight[fut] = idx
                    fut_device[fut] = dev
                    reserved[dev] += memory_estimates[idx]
                    pending.remove(idx)
                    logger.debug(
                        f"Shard {idx} -> {dev} "
                        f"(~{memory_estimates[idx] / 1e9:.2f} GB)"
                    )

                # --- nothing running and nothing fits ---
                if not inflight:
                    if not pending:
                        break
                    raise RuntimeError(
                        "No device has enough estimated free memory "
                        "for any remaining shard. Consider reducing "
                        "max_workers or adjusting _MEMORY_MULTIPLIER."
                    )

                # --- wait for at least one job to finish ---
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

    return results
