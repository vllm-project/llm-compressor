"""
Test that activation qparams (e.g. kv cache scales) are calibrated for modules which
are traced through rather than called by a sequential subgraph.

With `sequential_targets="Linear"`, attention modules are ancestors of the sequential
targets, so they are traced through and never passed to `on_sequential_epoch_end`.
Their activation qparams must still be updated before calibration ends.
"""

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs
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
def test_end_calibration_updates_traced_through_modules(
    sequential_targets, expect_attention, monkeypatch
):
    # record which modules have their activation qparams updated by end_calibration
    updated_at_end = []
    end_calibration = QuantizationMixin.end_calibration
    update_activation_qparams = QuantizationMixin.update_activation_qparams

    def spy_end_calibration(self, model):
        def spy_update(self, modules, **kwargs):
            updated_at_end.extend(modules)
            return update_activation_qparams(self, modules, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(QuantizationMixin, "update_activation_qparams", spy_update)
            return end_calibration(self, model)

    monkeypatch.setattr(QuantizationMixin, "end_calibration", spy_end_calibration)
    model = _calibrate_kv_cache(sequential_targets)

    # only modules which were not updated at a sequential epoch end are updated
    attention = [layer.self_attn for layer in model.model.layers]
    expected = attention if expect_attention else []
    assert len(updated_at_end) == len(expected)
    assert set(updated_at_end) == set(expected)


@pytest.mark.parametrize("already_updated", [True, False])
def test_end_calibration_only_updates_modules_once(already_updated, monkeypatch):
    modifier = QuantizationModifier(kv_cache_scheme=KV_CACHE_SCHEME)
    module = torch.nn.Linear(4, 4)
    module.input_observer = object()  # module has an activation observer
    if already_updated:
        modifier._activation_updated_modules.add(module)

    calls = []
    monkeypatch.setattr(
        QuantizationModifier,
        "update_activation_qparams",
        lambda _self, modules, **_: calls.append(modules),
    )
    monkeypatch.setattr(mixin, "match_named_modules", lambda *_: [("module", module)])
    monkeypatch.setattr(mixin, "freeze_module_quantization", lambda _: None)

    modifier.end_calibration(torch.nn.Sequential(module))

    # modules already updated at a sequential epoch end are not updated again, which
    # would otherwise double count observer statistics synced with `ReduceOp.SUM`
    assert calls == [[] if already_updated else [module]]
    assert modifier._activation_updated_modules == set()
