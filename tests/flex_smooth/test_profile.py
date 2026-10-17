# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from loguru import logger

from examples.glm52_precision.oneshot_profile import Profile
from llmcompressor.pipelines.sequential import pipeline


@pytest.mark.parametrize("mode,samples", [("summary", 0), ("sampled", 4), ("full", 10)])
def test_hot_sync_and_resource_sampling(monkeypatch, mode, samples):
    sync = Mock()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_device=lambda: 0, device_count=lambda: 1, synchronize=sync
        ),
        raising=False,
    )
    profile = Profile("npu", mode, sample_every=3)
    profile.memory = Mock(
        return_value={
            "cpu_rss_bytes": 10,
            "process_read_bytes": 0,
            "process_write_bytes": 0,
        }
    )
    for _ in range(10):
        with profile.stage("hot", hot=True):
            pass
    entry = profile.stages["hot"]
    assert entry["calls"] == 10
    assert entry.get("synchronized_samples", 0) == samples
    assert sync.call_count == profile.memory.call_count == samples * 2
    assert entry["timing_kind"] == (
        "synchronized_wall" if mode == "full" else "host_dispatch"
    )
    if mode != "full":
        assert entry["seconds"] == entry["host_dispatch_seconds"]
    with profile.stage("boundary"):
        pass
    assert sync.call_count == samples * 2 + 2
    assert profile.stages["boundary"]["synchronized_samples"] == 1
    json.dumps(profile.report)


@pytest.mark.parametrize("mode", ["summary", "sampled", "full"])
def test_profile_preserves_values_rng_and_exceptions(mode):
    profile = Profile("cpu", mode, sample_every=2)
    state = torch.get_rng_state().clone()
    values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    with profile.stage("hot", hot=True):
        result = values @ values.T
    torch.testing.assert_close(result, values @ values.T, rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())
    with pytest.raises(RuntimeError, match="original failure"):
        with profile.stage("hot", hot=True):
            raise RuntimeError("original failure")
    assert profile.stages["hot"]["calls"] == 2


def test_summary_progress_at_source_and_full_restoration(monkeypatch):
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    original = Mock(side_effect=lambda values, **kwargs: iter(values))
    monkeypatch.setattr(pipeline, "tqdm", original)
    try:
        with Profile("cpu").instrument():
            for index in range(1, 101):
                for phase in ("Calibrating", "Propagating"):
                    assert list(
                        pipeline.tqdm([1, 2], desc=f"({index}/100): {phase}")
                    ) == [1, 2]
            logger.warning("warning survives")
            logger.error("error survives")
        assert pipeline.tqdm is original
        original.assert_not_called()
        assert sum("[GLM calibration]" in m for m in messages) == 21
        assert any("completed 100/100" in m for m in messages)
        assert any("warning survives" in m for m in messages)
        assert any("error survives" in m for m in messages)
        with Profile("cpu", "full").instrument():
            assert list(pipeline.tqdm([3], desc="(1/1): Calibrating")) == [3]
        original.assert_called_once()
        with pytest.raises(ValueError, match="propagation failed"):
            with Profile("cpu").instrument():
                for _ in pipeline.tqdm([0], desc="(1/1): Propagating"):
                    raise ValueError("propagation failed")
        assert pipeline.tqdm is original
    finally:
        logger.remove(sink)


def test_non_source_progress_is_quiet(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    log = Mock()
    monkeypatch.setattr(logger, "info", log)
    original = Mock()
    wrapped = Profile("cpu").progress(original)
    assert list(wrapped([1], desc="(1/1): Propagating")) == [1]
    original.assert_not_called()
    log.assert_not_called()
