"""
Unit tests for the dynamic shard scheduler.

These run without a GPU and cover the core scheduling logic with mocked
device memory, so they execute on every CI run — not just multi-GPU.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from llmcompressor.entrypoints.model_free.scheduler import (
    _empty_cache,
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


# ── _pick_device ────────────────────────────────────────────────────────


def _mock_gdm(free_map):
    """Build a mock torch.get_device_module that returns per-device free
    memory from *free_map*."""

    def _factory(dev):
        m = MagicMock()
        m.mem_get_info.return_value = (
            free_map.get(dev, 0),
            96_000_000_000,
        )
        return m

    return _factory


@patch("torch.get_device_module")
def test_pick_most_free_device(mock_gdm):
    d0 = torch.device("cuda:0")
    d1 = torch.device("cuda:1")
    mock_gdm.side_effect = _mock_gdm({d0: 40_000_000_000, d1: 80_000_000_000})
    reserved = {d0: 0, d1: 0}
    assert _pick_device([d0, d1], 1000, reserved) == d1


@patch("torch.get_device_module")
def test_pick_none_when_nothing_fits(mock_gdm):
    d0 = torch.device("cuda:0")
    mock_gdm.side_effect = _mock_gdm({d0: 1000})
    assert _pick_device([d0], 2000, {d0: 0}) is None


@patch("torch.get_device_module")
def test_pick_respects_reservations(mock_gdm):
    d0 = torch.device("cuda:0")
    d1 = torch.device("cuda:1")
    mock_gdm.side_effect = _mock_gdm({d0: 80_000_000_000, d1: 50_000_000_000})
    # d0: 80 GB free - 70 GB reserved = 10 GB available
    # d1: 50 GB free - 0 reserved = 50 GB available
    # job needs 20 GB -> pick d1
    reserved = {d0: 70_000_000_000, d1: 0}
    assert _pick_device([d0, d1], 20_000_000_000, reserved) == d1


def test_pick_cpu_always():
    """CPU is always eligible regardless of memory."""
    cpu = torch.device("cpu")
    picked = _pick_device([cpu], 10**15, {cpu: 0})
    assert picked == cpu


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


# ── _empty_cache ───────────────────────────────────────────────────────


def test_empty_cache_noop_on_cpu():
    """_empty_cache should do nothing on CPU devices."""
    _empty_cache(torch.device("cpu"))  # should not raise


# ── _free_bytes ────────────────────────────────────────────────────────


@patch("torch.get_device_module")
def test_free_bytes_missing_mem_get_info(mock_gdm):
    """Backend without mem_get_info should raise RuntimeError."""
    mock_module = MagicMock(spec=[])  # no attributes
    mock_gdm.return_value = mock_module
    dev = torch.device("cuda:0")
    with pytest.raises(RuntimeError, match="does not support mem_get_info"):
        _free_bytes(dev, {dev: 0})
