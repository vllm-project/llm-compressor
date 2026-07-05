"""
Tests for static KV-cache calibration scale validation.

KV-cache quantization relies on key/value observers collecting calibration
statistics before k_scale/v_scale are saved. These tests use small CPU-only
stubs so the calibration lifecycle can be validated without downloading a model.
"""

import pytest
import torch
import torch.nn as nn
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStatus,
    apply_quantization_config,
    is_attention_module,
)
from transformers import PretrainedConfig

from llmcompressor.modifiers.quantization.calibration import observe, update_qparams
from llmcompressor.modifiers.quantization.quantization import QuantizationModifier


class _StubAttention(nn.Module):
    """Minimal attention module recognized by is_attention_module()."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class _StubBlock(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        # Named "attention" instead of "self_attn" to exercise name-agnostic
        # attention module discovery.
        self.attention = _StubAttention(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        return self.mlp(self.attention(x))


class _StubModel(nn.Module):
    def __init__(self, dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.config = PretrainedConfig(
            num_attention_heads=1,
            num_key_value_heads=1,
            hidden_size=dim,
        )
        self.layers = nn.ModuleList([_StubBlock(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _quant_args(dynamic: bool | str = False):
    strategy = "tensor_group" if dynamic == "local" else "tensor"
    group_size = 16 if dynamic == "local" else None

    return QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=strategy,
        group_size=group_size,
        dynamic=dynamic,
        symmetric=True,
    )


def _kv_cache_modifier(dynamic: bool | str = False, ignore: list[str] | None = None):
    return QuantizationModifier(
        targets=["Linear"],
        ignore=ignore or [],
        kv_cache_scheme=_quant_args(dynamic=dynamic),
    )


def _attention_modules(model):
    return [(name, m) for name, m in model.named_modules() if is_attention_module(m)]


def _prepare_kv_model(
    dim: int = 16,
    num_layers: int = 2,
    dynamic: bool | str = False,
    ignore: list[str] | None = None,
):
    model = _StubModel(dim=dim, num_layers=num_layers)
    modifier = _kv_cache_modifier(dynamic=dynamic, ignore=ignore)
    attn_modules = _attention_modules(model)

    apply_quantization_config(model, modifier.resolved_config)
    modifier.start_calibration(model)

    return model, modifier, attn_modules


def _observe_weight_observers(model):
    for module in model.modules():
        if getattr(module, "weight_observer", None) is not None:
            observe(module, "weight")
            update_qparams(module, "weight")


def _observe_kv_cache(attn_modules, dim: int = 16):
    key_states = torch.ones(1, 1, 1, dim)
    value_states = torch.full((1, 1, 1, dim), 2.0)

    for _, module in attn_modules:
        module.k_observer(key_states)
        module.v_observer(value_states)
        update_qparams(module, ("k", "v"))


def test_attention_module_not_named_self_attn_gets_calibrated():
    """Attention modules named 'attention' should still get KV observers/hooks."""
    model, modifier, attn_modules = _prepare_kv_model(dim=16)

    assert len(attn_modules) == 2, "Expected 2 attention modules in _StubModel"
    for name, module in attn_modules:
        assert "self_attn" not in name, f"{name} unexpectedly matches self_attn"
        assert hasattr(module, "quantization_scheme"), (
            "apply_quantization_config should set quantization_scheme on attention "
            "modules"
        )
        assert module.quantization_status == QuantizationStatus.CALIBRATION, (
            f"Attention module '{name}' was not calibrated "
            f"(status={module.quantization_status})"
        )

    assert len(modifier._calibration_hooks) > 0, (
        "Expected calibration hooks to be registered"
    )

    _observe_weight_observers(model)
    _observe_kv_cache(attn_modules)
    modifier.end_calibration(model)

    for name, module in attn_modules:
        assert module.quantization_status == QuantizationStatus.FROZEN, (
            f"Attention module '{name}' was not frozen after end_calibration"
        )


def test_static_kv_cache_calibration_requires_observed_kv_tensors():
    model, modifier, attn_modules = _prepare_kv_model(dim=16, num_layers=1)
    _observe_weight_observers(model)

    with pytest.raises(ValueError) as exc_info:
        modifier.end_calibration(model)

    error_message = str(exc_info.value)
    assert "KV-cache quantization calibration failed" in error_message
    assert "layers.0.attention.k_observer" in error_message
    assert "layers.0.attention.v_observer" in error_message
    assert "update(...) API" in error_message

    attention = attn_modules[0][1]
    assert attention.quantization_status == QuantizationStatus.CALIBRATION
    assert hasattr(attention, "k_observer")
    assert hasattr(attention, "v_observer")


@pytest.mark.parametrize("scale_name", ["k_scale", "v_scale"])
def test_static_kv_cache_calibration_rejects_zero_scales(scale_name):
    model, modifier, attn_modules = _prepare_kv_model(dim=16, num_layers=1)
    _observe_weight_observers(model)
    _observe_kv_cache(attn_modules)
    getattr(attn_modules[0][1], scale_name).data.zero_()

    with pytest.raises(ValueError) as exc_info:
        modifier.end_calibration(model)

    error_message = str(exc_info.value)
    assert f"layers.0.attention.{scale_name}" in error_message
    assert "non-positive" in error_message


@pytest.mark.parametrize("scale_name", ["k_scale", "v_scale"])
def test_static_kv_cache_calibration_rejects_missing_scales(scale_name):
    model, modifier, attn_modules = _prepare_kv_model(dim=16, num_layers=1)
    _observe_weight_observers(model)
    _observe_kv_cache(attn_modules)
    delattr(attn_modules[0][1], scale_name)

    with pytest.raises(ValueError) as exc_info:
        modifier.end_calibration(model)

    error_message = str(exc_info.value)
    assert f"layers.0.attention.{scale_name}" in error_message
    assert "missing" in error_message


def test_static_kv_cache_calibration_rejects_non_finite_scales():
    model, modifier, attn_modules = _prepare_kv_model(dim=16, num_layers=1)
    _observe_weight_observers(model)
    _observe_kv_cache(attn_modules)
    attn_modules[0][1].v_scale.data.fill_(float("nan"))

    with pytest.raises(ValueError) as exc_info:
        modifier.end_calibration(model)

    error_message = str(exc_info.value)
    assert "layers.0.attention.v_scale" in error_message
    assert "non-finite" in error_message


def test_static_kv_cache_calibration_respects_ignore_list():
    model, modifier, attn_modules = _prepare_kv_model(
        dim=16,
        num_layers=2,
        ignore=["layers.1.attention"],
    )
    _observe_weight_observers(model)
    _observe_kv_cache([attn_modules[0]])
    attn_modules[1][1].k_scale.data.zero_()
    attn_modules[1][1].v_scale.data.zero_()

    modifier.end_calibration(model)


def test_dynamic_kv_cache_calibration_skips_static_scale_validation():
    model, modifier, _ = _prepare_kv_model(dim=16, num_layers=1, dynamic=True)
    _observe_weight_observers(model)

    modifier.end_calibration(model)


def test_local_dynamic_kv_cache_calibration_skips_static_scale_validation():
    model, modifier, _ = _prepare_kv_model(dim=16, num_layers=1, dynamic="local")
    _observe_weight_observers(model)

    modifier.end_calibration(model)


def test_end_calibration_can_run_after_static_kv_cache_is_frozen():
    model, modifier, attn_modules = _prepare_kv_model(dim=16, num_layers=1)
    _observe_weight_observers(model)
    _observe_kv_cache(attn_modules)

    modifier.end_calibration(model)
    attn_modules[0][1].k_scale.data.zero_()
    modifier.end_calibration(model)

    assert attn_modules[0][1].quantization_status == QuantizationStatus.FROZEN


def test_non_kv_quantization_flow_is_not_validated():
    model = nn.Sequential(nn.Linear(4, 4))
    modifier = QuantizationModifier(targets=["Linear"])

    apply_quantization_config(model, modifier.resolved_config)
    modifier.start_calibration(model)
    modifier.end_calibration(model)

    assert model[0].quantization_status == QuantizationStatus.FROZEN
