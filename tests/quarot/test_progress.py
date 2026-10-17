# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""QuaRot observability must preserve fusion/reset and rotation execution order."""

import importlib
import os
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llmcompressor.core import EventType, State
from llmcompressor.core.lifecycle import CompressionLifecycle
from llmcompressor.modifiers.transform.quarot import base

from .tiny_glm import TinyGLM


@torch.no_grad()
def test_progress_preserves_numerics_consumers_resets_and_operations(monkeypatch):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = TinyGLM().double().eval()
    expected = deepcopy(model)
    reference = base.QuaRotModifier(block_size=32, precision="float64")
    reference.initialize(State(model=expected))
    # Execute the original loops without progress wrappers as an exact baseline.
    for fusion in reference._plan.fusions:
        base.fuse_norm_linears(
            expected.get_submodule(fusion.norm),
            [expected.get_submodule(name) for name in fusion.consumers],
            precision=reference.precision,
        )
    for operation in (reference._plan.embedding, *reference._plan.rotations):
        reference._rotate(expected, operation)

    lifecycle = CompressionLifecycle()
    lifecycle.initialize(
        model=model,
        recipe=[base.QuaRotModifier(block_size=32, precision="float64")],
    )
    plan = lifecycle.recipe.modifiers[0]._plan
    names = {module: name for name, module in model.named_modules()}
    consumer_updates, norm_resets, fusion_calls, rotations, logs = [], [], [], [], []
    fuse_module = importlib.import_module("llmcompressor.modeling.fuse")
    original_update = fuse_module.update_offload_parameter
    original_fuse = base.fuse_norm_linears
    original_rotate = base.QuaRotModifier._rotate

    def update(module, key, value):
        fusion = plan.fusions[len(fusion_calls) - 1]
        name = names[module]
        if name == fusion.norm:
            # Every consumer write has completed before the norm is reset.
            assert consumer_updates[-len(fusion.consumers) :] == list(fusion.consumers)
            assert not torch.equal(module.weight, torch.ones_like(module.weight))
            norm_resets.append(name)
        else:
            assert name in fusion.consumers
        original_update(module, key, value)
        if name != fusion.norm:
            consumer_updates.append(name)

    def fuse(norm, consumers, precision):
        fusion = plan.fusions[len(fusion_calls)]
        assert names[norm] == fusion.norm
        gain, visited = norm.weight.clone(), []
        fusion_calls.append(fusion.norm)

        def checked_consumers():
            for consumer in consumers:
                assert torch.equal(norm.weight, gain)
                assert names[consumer] == fusion.consumers[len(visited)]
                visited.append(names[consumer])
                yield consumer
                assert consumer_updates[-1] == names[consumer]

        original_fuse(norm, checked_consumers(), precision=precision)
        assert visited == list(fusion.consumers)
        assert torch.equal(norm.weight, torch.ones_like(gain))

    def rotate(modifier, model, operation):
        original_rotate(modifier, model, operation)
        rotations.append(operation)

    def log(template, *args):
        message = template.format(*args)
        logs.append(message)
        match = re.match(r"\[QuaRot\] (norm fusion|rotation) (\d+)/(\d+) \|", message)
        if match:
            completed = consumer_updates if match[1] == "norm fusion" else rotations
            assert int(match[2]) == len(completed)
        if message.startswith("[QuaRot] norm fusion complete:"):
            assert norm_resets == [fusion.norm for fusion in plan.fusions]

    monkeypatch.setattr(fuse_module, "update_offload_parameter", update)
    monkeypatch.setattr(base, "fuse_norm_linears", fuse)
    monkeypatch.setattr(base.QuaRotModifier, "_rotate", rotate)
    monkeypatch.setattr(base, "logger", SimpleNamespace(info=log))
    lifecycle.event(EventType.CALIBRATION_START)
    assert fusion_calls == norm_resets == [fusion.norm for fusion in plan.fusions]
    assert consumer_updates == [
        name for fusion in plan.fusions for name in fusion.consumers
    ]
    assert rotations == [plan.embedding, *plan.rotations]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
    phases = ["preparation / validation", "norm fusion", "rotation", "total"]
    completions = [message for message in logs if " complete:" in message]
    assert len(completions) == 4
    for phase, message in zip(phases, completions, strict=True):
        assert message.startswith(f"[QuaRot] {phase} complete:")
        assert float(message.split(": ")[1].split()[0]) >= 0
    log_count = len(logs)
    lifecycle.event(EventType.CALIBRATION_START)
    assert len(logs) == log_count  # Already applied: no duplicate work or progress.
    lifecycle.event(EventType.CALIBRATION_END)
    lifecycle.finalize()
    if output := os.environ.get("QUAROT_REPORT_DIR"):
        Path(output, "quarot-progress.log").write_text(
            "\n".join(logs), encoding="utf-8"
        )


def test_progress_is_throttled_and_does_not_count_failed_work(monkeypatch):
    logs = []
    monkeypatch.setattr(
        base,
        "logger",
        SimpleNamespace(
            info=lambda template, *args: logs.append(template.format(*args))
        ),
    )
    progress = base._QuaRotProgress("rotation", 1001)
    assert list(progress.track(range(1001))) == list(range(1001))
    progress.complete()
    assert progress.completed == 1001
    assert len(logs) <= 22  # Start, approximately 5% steps, and phase completion.
    assert any("rotation 1001/1001" in message for message in logs)

    logs.clear()
    progress = base._QuaRotProgress("norm fusion", 1001)
    with pytest.raises(OSError, match="write failed"):
        for item in progress.track(range(1001)):
            if item == 73:
                raise OSError("write failed")
    assert progress.completed == 73
    assert not any("complete:" in message or "1001/1001" in message for message in logs)
