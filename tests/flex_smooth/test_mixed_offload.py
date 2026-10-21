# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact resident/offloaded comparisons for the complete mixed recipe lifecycle."""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from compressed_tensors.offload import OffloadCache, disable_offloading, offload_module

from llmcompressor.core import EventType
from llmcompressor.core.lifecycle import CompressionLifecycle
from llmcompressor.recipe import Recipe
from llmcompressor.utils.helpers import DisableQuantization
from tests.quarot.tiny_glm import TinyGLM

TOKENS = torch.tensor([[1, 3, 7, 9], [9, 1, 5, 2]])


def calibrate(model):
    recipe = Recipe.create_instance(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/glm52_precision/mixed_mxfp.yaml"
        )
    )
    life = CompressionLifecycle()
    life.initialize(model=model, recipe=recipe)
    life.event(EventType.CALIBRATION_START)
    with DisableQuantization(model):
        model(TOKENS)
    life.event(EventType.SEQUENTIAL_EPOCH_END, modules=list(model.modules()))
    life.event(EventType.CALIBRATION_END)
    life.finalize()
    flex, quant = recipe.modifiers[1:]
    assert not flex._hooks and not flex._cache
    assert not quant._calibration_hooks
    return flex.diagnostics


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "disk"])
@torch.no_grad()
def test_complete_mixed_recipe_offload(tmp_path, dtype, device):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = TinyGLM().to(dtype).eval()
    baseline = deepcopy(model)
    expected_diagnostics = calibrate(baseline)
    expected_output = baseline(TOKENS)[0]
    for module in model.modules():
        if list(module.parameters(recurse=False)):
            offload_module(
                module,
                onload_device="cpu",
                offload_device=device,
                **({"offload_dir": tmp_path} if device == "disk" else {}),
            )
    actual_diagnostics = calibrate(model)
    expected_state, actual_state = baseline.state_dict(), model.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, value in actual_state.items():
        torch.testing.assert_close(value, expected_state[name], atol=0, rtol=0)
    for name, expected in expected_diagnostics.items():
        actual = actual_diagnostics[name]
        torch.testing.assert_close(
            actual.pop("scale"), expected.pop("scale"), atol=0, rtol=0
        )
        assert actual == expected
    # Match the sequential pipeline's cache lifetime. A bare DiskCache forward
    # reloads weight on each attribute access and loses CT's temporary weight Q/DQ.
    # This context is bounded by the tiny diagnostic forward, not a full-model recipe.
    with disable_offloading():
        torch.testing.assert_close(model(TOKENS)[0], expected_output, atol=0, rtol=0)
    assert not OffloadCache.keep_onloaded_values
    module = model.model.layers[0].self_attn.q_a_proj
    assert isinstance(module._parameters, OffloadCache)
    assert str(module._parameters.offload_device) == device
    if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        Path(output, f"mixed-offload-{device}-{dtype}.json").write_text(
            json.dumps(
                {
                    "real_weights": False,
                    "dtype": str(dtype),
                    "offload_device": device,
                    "state_tensors_compared": len(actual_state),
                    "state_max_abs_error": 0,
                    "logits_max_abs_error": 0,
                    "search_and_scales_exact": True,
                    "forward_context": "disable_offloading, as in sequential pipeline",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
