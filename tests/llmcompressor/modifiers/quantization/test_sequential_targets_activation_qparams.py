"""
Test that quantization parameters (e.g. kv cache scales) are calibrated for modules
which are traced through rather than called by a sequential subgraph.

With `sequential_targets="Linear"`, attention modules are ancestors of the sequential
targets, so they are traced through and never passed to `on_sequential_epoch_end`.
Their quantization parameters must still be updated before calibration ends.
"""

import io

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs
from loguru import logger
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin, mixin

MODEL_ID = "nm-testing/tinysmokellama-3.2"
KV_CACHE_SCHEME = QuantizationArgs(
    num_bits=8, type="float", strategy="tensor", symmetric=True
)


def _calibrate_kv_cache(sequential_targets: str) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    oneshot(
        model=model,
        dataset="open_platypus",
        splits={"calibration": "train[:4]"},
        num_calibration_samples=4,
        shuffle_calibration_samples=False,
        recipe=QuantizationModifier(kv_cache_scheme=KV_CACHE_SCHEME),
        sequential_targets=sequential_targets,
    )
    return model


def _kv_scales(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    attention = [layer.self_attn for layer in model.model.layers]
    k_scales = torch.stack([attn.k_scale.detach().float() for attn in attention])
    v_scales = torch.stack([attn.v_scale.detach().float() for attn in attention])
    return k_scales, v_scales


def test_kv_cache_scales_with_linear_sequential_targets():
    # with decoder layer targets, attention is called by a subgraph
    ref_k_scales, ref_v_scales = _kv_scales(_calibrate_kv_cache("LlamaDecoderLayer"))
    assert torch.isfinite(ref_k_scales).all() and (ref_k_scales > 0).all()
    assert torch.isfinite(ref_v_scales).all() and (ref_v_scales > 0).all()

    # with linear targets, attention is traced through. Weights are not quantized,
    # so both runs observe the same activations and must produce the same scales, up
    # to numerical differences between attention implementations
    k_scales, v_scales = _kv_scales(_calibrate_kv_cache("Linear"))
    assert torch.allclose(k_scales, ref_k_scales, rtol=1e-4, atol=0)
    assert torch.allclose(v_scales, ref_v_scales, rtol=1e-4, atol=0)


@pytest.mark.parametrize(
    "sequential_targets,expect_attention",
    [("LlamaDecoderLayer", False), ("Linear", True)],
)
def test_end_calibration_quantizes_remaining_modules(
    sequential_targets, expect_attention, monkeypatch
):
    # record modules passed to `on_sequential_epoch_end` by `end_calibration`
    remaining = []
    end_calibration = QuantizationMixin.end_calibration
    on_sequential_epoch_end = QuantizationModifier.on_sequential_epoch_end

    def spy_end_calibration(self, model):
        def spy_epoch_end(self, state, event, modules, **kwargs):
            remaining.extend(modules)
            return on_sequential_epoch_end(self, state, event, modules, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(
                QuantizationModifier, "on_sequential_epoch_end", spy_epoch_end
            )
            return end_calibration(self, model)

    monkeypatch.setattr(QuantizationMixin, "end_calibration", spy_end_calibration)

    log_output = io.StringIO()
    handler_id = logger.add(log_output, level="WARNING")
    try:
        model = _calibrate_kv_cache(sequential_targets)
    finally:
        logger.remove(handler_id)

    # only modules which were not called by a sequential subgraph are quantized
    # at the end of calibration, and the user is warned about them
    attention = [layer.self_attn for layer in model.model.layers]
    expected = attention if expect_attention else []
    assert len(remaining) == len(expected)
    assert set(remaining) == set(expected)

    warning = "which were calibrated but not quantized"
    assert (warning in log_output.getvalue()) == expect_attention


class _Observer:
    def __init__(self, has_statistics: bool):
        self.has_statistics = has_statistics


@pytest.mark.parametrize(
    "observers,expected",
    [
        ({}, False),
        ({"weight": False, "input": False}, False),
        ({"weight": True}, True),
        ({"input": True}, True),
        ({"k": False, "v": True}, True),
    ],
)
def test_has_observer_statistics(observers, expected):
    module = torch.nn.Linear(4, 4)
    for base_name, has_statistics in observers.items():
        setattr(module, f"{base_name}_observer", _Observer(has_statistics))

    assert mixin._has_observer_statistics(module) == expected
