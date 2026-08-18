"""
Unit tests for the dynamic shard scheduler.

These run without a GPU and cover the core scheduling logic with mocked
device memory, so they execute on every CI run — not just multi-GPU.
"""

from unittest.mock import patch

import pytest
import torch

from llmcompressor.entrypoints.model_free.scheduler import (
    _free_bytes,
    _pick_device,
    estimate_job_memory,
    exec_jobs_dynamic,
)

# ── estimate_job_memory ─────────────────────────────────────────────────


def test_estimate_sums_file_sizes(tmp_path):
    """Memory estimate = sum(file sizes) * multiplier."""
    f1 = tmp_path / "a.safetensors"
    f2 = tmp_path / "b.safetensors"
    f1.write_bytes(b"\x00" * 1000)
    f2.write_bytes(b"\x00" * 2000)
    iwm = {str(f1): None, str(f2): None}
    assert estimate_job_memory(iwm) == int(3000 * 3.0)


def test_estimate_single_file(tmp_path):
    f = tmp_path / "shard.safetensors"
    f.write_bytes(b"\x00" * 500)
    assert estimate_job_memory({str(f): None}) == int(500 * 3.0)


# ── _pick_device (no mocks needed — initial_free is a plain dict) ──────


def test_pick_most_free_device():
    d0 = torch.device("cuda:0")
    d1 = torch.device("cuda:1")
    initial_free = {d0: 40_000_000_000, d1: 80_000_000_000}
    reserved = {d0: 0, d1: 0}
    assert _pick_device([d0, d1], 1000, initial_free, reserved) == d1


def test_pick_none_when_nothing_fits():
    d0 = torch.device("cuda:0")
    initial_free = {d0: 1000}
    assert _pick_device([d0], 2000, initial_free, {d0: 0}) is None


def test_pick_respects_reservations():
    d0 = torch.device("cuda:0")
    d1 = torch.device("cuda:1")
    initial_free = {d0: 80_000_000_000, d1: 50_000_000_000}
    # d0: 80 GB initial - 70 GB reserved = 10 GB available
    # d1: 50 GB initial - 0 reserved = 50 GB available
    # job needs 20 GB -> pick d1
    reserved = {d0: 70_000_000_000, d1: 0}
    assert _pick_device([d0, d1], 20_000_000_000, initial_free, reserved) == d1


def test_pick_skips_cpu_devices():
    """CPU devices are not in initial_free, so they are never picked."""
    cpu = torch.device("cpu")
    picked = _pick_device([cpu], 1000, {}, {cpu: 0})
    assert picked is None


# ── _free_bytes ────────────────────────────────────────────────────────


def test_free_bytes_subtracts_reserved():
    dev = torch.device("cuda:0")
    initial_free = {dev: 10_000_000_000}
    reserved = {dev: 3_000_000_000}
    assert _free_bytes(dev, initial_free, reserved) == 7_000_000_000


def test_free_bytes_clamps_to_zero():
    """Should never return negative even if reserved exceeds initial."""
    dev = torch.device("cuda:0")
    initial_free = {dev: 1000}
    reserved = {dev: 5000}
    assert _free_bytes(dev, initial_free, reserved) == 0


def test_free_bytes_unknown_device_returns_zero():
    """Devices not in initial_free (e.g. CPU) get zero available."""
    cpu = torch.device("cpu")
    assert _free_bytes(cpu, {}, {}) == 0


# ── exec_jobs_dynamic (CPU path, no GPU needed) ────────────────────────


def test_cpu_path_runs_all_jobs():
    def fn(iwm, sp, sch, ign, dev, conv):
        return (100, {"t": sp})

    jobs = [(fn, {}, f"s{i}.st", "FP8", [], None) for i in range(5)]
    mem = [1000] * 5
    out = exec_jobs_dynamic(jobs, [torch.device("cpu")], 2, mem, desc="Test")
    assert len(out) == 5
    assert all(r[0] == 100 for r in out)


def test_cpu_path_empty_jobs():
    out = exec_jobs_dynamic([], [torch.device("cpu")], 1, [], desc="Test")
    assert out == []


def test_cpu_path_preserves_order():
    """Results should match job order, not completion order."""

    def fn(iwm, sp, sch, ign, dev, conv):
        return (int(sp), {})

    jobs = [(fn, {}, str(i), "s", [], None) for i in range(10)]
    mem = [100] * 10
    out = exec_jobs_dynamic(jobs, [torch.device("cpu")], 4, mem, desc="Test")
    assert [r[0] for r in out] == list(range(10))


# ── exec_jobs_dynamic error handling ──────────────────────────────────


@patch(
    "llmcompressor.entrypoints.model_free.scheduler"
    ".torch.accelerator.memory.get_memory_info"
)
def test_raises_when_no_device_fits(mock_mem_info):
    """Multi-worker path should raise RuntimeError when no device can fit
    any remaining shard and nothing is in flight."""
    mock_mem_info.return_value = (1000, 96_000_000_000)

    def fn(iwm, sp, sch, ign, dev, conv):
        return (0, {})

    # job needs 10 GB but device only has 1000 bytes
    jobs = [(fn, {}, "s0.st", "FP8", [], None)]
    mem = [10_000_000_000]

    with pytest.raises(RuntimeError, match="No device has enough"):
        exec_jobs_dynamic(
            jobs,
            [torch.device("cuda:0")],
            2,
            mem,
            desc="Test",
        )
