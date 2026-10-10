# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from examples.glm52_precision.oneshot_profile import Profile
from examples.glm52_precision.profile_attribution import PhaseAttribution
from llmcompressor.modifiers.transform.quarot.base import _QuaRotProgress


def test_attribution_disabled_is_noop():
    profile = Profile("cpu")
    before = torch.nn.Module.register_parameter
    with profile.attribution_scope("save"):
        assert torch.nn.Module.register_parameter is before
    assert "attribution" not in profile.report


def test_real_qparam_creation_and_async_collective_return(monkeypatch):
    from compressed_tensors.quantization import QuantizationArgs

    init = import_module("compressed_tensors.quantization.lifecycle.initialize")
    parameter = init.Parameter
    register = torch.nn.Module.register_parameter
    npu_sync = Mock()
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=npu_sync), raising=False
    )
    work = object()
    reduce = Mock(return_value=work)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    module = torch.nn.Linear(4, 4)
    report = {}
    probe = PhaseAttribution(report)
    with probe.instrument("initialization"):
        init.initialize_qparams(
            module, "weight", QuantizationArgs(num_bits=8), (4, 4), torch.float32
        )
        assert isinstance(module.weight_scale, torch.nn.Parameter)
        assert torch.distributed.all_reduce(module.weight_scale, async_op=True) is work
    npu_sync.assert_not_called()
    assert init.Parameter is parameter
    assert torch.nn.Module.register_parameter is register
    assert torch.distributed.all_reduce is reduce
    assert probe.operations["initialization/initialize_qparams"]["calls"] == 1
    assert probe.operations["initialization/qparam_parameter_construct"]["calls"] == 2
    assert probe.operations["initialization/register_parameter"]["calls"] == 2
    json.dumps(report)


def test_progress_cache_segments_conserve_jobs_bytes_and_failures():
    report = {}
    probe = PhaseAttribution(report)
    seen = []
    value = torch.zeros(4)
    original = Mock(side_effect=lambda cache, tensor: tensor)
    onload = probe.cache_call(original, "CPUCache.onload")
    progress = _QuaRotProgress("rotation", 100)
    track = probe.progress(_QuaRotProgress.track)
    for item in track(progress, range(100)):
        seen.append(item)
        assert onload(None, value) is value
    assert seen == list(range(100))
    assert progress.completed == 100
    assert len(probe.segments) == 20
    assert (
        sum(
            s["cache_delta"]["quarot.rotation/CPUCache.onload"]["calls"]
            for s in probe.segments
        )
        == 100
    )
    assert (
        sum(
            s["cache_delta"]["quarot.rotation/CPUCache.onload"]["logical_bytes"]
            for s in probe.segments
        )
        == 1600
    )
    assert probe.phase.get() == "outside"
    bad = probe.cache_call(
        Mock(side_effect=ValueError("original error")), "DiskCache.onload"
    )
    with pytest.raises(ValueError, match="original error"):
        bad(None, value)
    assert probe.caches["outside/DiskCache.onload"]["failures"] == 1


def test_save_state_dict_root_and_context_restore_on_error():
    module = torch.nn.Sequential(torch.nn.Linear(2, 2))
    original = torch.nn.Module.state_dict
    expected = module.state_dict()
    probe = PhaseAttribution({})
    with pytest.raises(RuntimeError, match="save failure"):
        with probe.instrument("save"):
            actual = module.state_dict()
            assert actual.keys() == expected.keys()
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
            raise RuntimeError("save failure")
    assert torch.nn.Module.state_dict is original
    assert probe.operations["save/state_dict_root"]["calls"] == 1
    assert probe.operations["save/total"]["failures"] == 1


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="No local NPU; server acceptance required",
)
def test_npu_profile_boundary_smoke():
    profile = Profile("npu")
    value = torch.arange(16, dtype=torch.float32, device="npu").reshape(4, 4)
    with profile.stage("tiny_boundary"):
        with profile.stage("tiny_hot", hot=True):
            output = value @ value.T
    torch.testing.assert_close(output.cpu(), value.cpu() @ value.cpu().T)
    assert profile.stages["tiny_hot"].get("synchronized_samples", 0) == 0
