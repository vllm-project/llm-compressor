"""
Tests for KV-cache calibration observer validation.

KV-cache quantization relies on key/value observers collecting calibration
statistics before quantization parameters are updated. These tests use small
CPU-only stubs so the calibration lifecycle can be validated without downloading
a model.
"""

import pytest
import torch
import torch.nn as nn
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    is_attention_module,
)
from transformers import PretrainedConfig

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.quantization.quantization import QuantizationModifier


class _StubAttention(nn.Module):
    """Minimal attention module recognized by is_attention_module()."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.config = PretrainedConfig(
            num_attention_heads=1,
            num_key_value_heads=1,
            hidden_size=dim,
        )

    def forward(self, x):
        return x


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
        self.lm_head = nn.Linear(dim, dim)

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        pass

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


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
    state = State(model=model)
    attn_modules = _attention_modules(model)

    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)

    return model, modifier, state, attn_modules


def _observe_kv_cache(attn_modules, dim: int = 16):
    key_states = torch.ones(1, 1, 1, dim)
    value_states = torch.full((1, 1, 1, dim), 2.0)

    for _, module in attn_modules:
        module.k_observer(key_states)
        module.v_observer(value_states)


def _sequential_epoch_end(modifier, state, model):
    modifier.on_sequential_epoch_end(
        state,
        Event(type_=EventType.SEQUENTIAL_EPOCH_END),
        modules=list(model.modules()),
    )


def test_attention_module_not_named_self_attn_gets_calibrated():
    """Attention modules named 'attention' should still get KV observers/hooks."""
    model, modifier, state, attn_modules = _prepare_kv_model(dim=16)

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
        assert hasattr(module, "k_observer")
        assert hasattr(module, "v_observer")

    assert len(modifier._calibration_hooks) > 0, (
        "Expected calibration hooks to be registered"
    )

    _observe_kv_cache(attn_modules)
    _sequential_epoch_end(modifier, state, model)
    modifier.end_calibration(model)

    for name, module in attn_modules:
        assert module.quantization_status == QuantizationStatus.FROZEN, (
            f"Attention module '{name}' was not frozen after end_calibration"
        )


def test_static_kv_cache_calibration_requires_observed_kv_tensors():
    model, modifier, state, attn_modules = _prepare_kv_model(dim=16, num_layers=1)

    with pytest.raises(ValueError) as exc_info:
        _sequential_epoch_end(modifier, state, model)

    error_message = str(exc_info.value)
    assert "Quantization calibration failed" in error_message
    assert "layers.0.attention.k_observer" in error_message
    assert "layers.0.attention.v_observer" in error_message

    attention = attn_modules[0][1]
    assert attention.quantization_status == QuantizationStatus.CALIBRATION
    assert hasattr(attention, "k_observer")
    assert hasattr(attention, "v_observer")


def test_static_kv_cache_calibration_uses_observer_counts():
    _, _, _, attn_modules = _prepare_kv_model(dim=16, num_layers=1)
    attention = attn_modules[0][1]

    assert attention.k_observer.num_observations == 0
    assert attention.v_observer.num_observations == 0

    _observe_kv_cache(attn_modules)

    assert attention.k_observer.num_observations == 1
    assert attention.v_observer.num_observations == 1


def test_static_kv_cache_calibration_respects_ignore_list():
    model, modifier, state, attn_modules = _prepare_kv_model(
        dim=16,
        num_layers=2,
        ignore=["layers.1.attention"],
    )
    _observe_kv_cache([attn_modules[0]])

    assert not hasattr(attn_modules[1][1], "k_observer")
    assert not hasattr(attn_modules[1][1], "v_observer")

    _sequential_epoch_end(modifier, state, model)


def test_dynamic_kv_cache_calibration_without_static_observers_passes():
    model, modifier, state, attn_modules = _prepare_kv_model(
        dim=16,
        num_layers=1,
        dynamic=True,
    )

    assert not hasattr(attn_modules[0][1], "k_observer")
    assert not hasattr(attn_modules[0][1], "v_observer")

    _sequential_epoch_end(modifier, state, model)


def test_non_kv_quantization_flow_is_validated_and_frozen():
    model = nn.Sequential(nn.Linear(4, 4))
    weight_args = QuantizationArgs(
        num_bits=8, type="int", symmetric=True, strategy="tensor"
    )
    modifier = QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(targets=["Linear"], weights=weight_args)
        }
    )
    state = State(model=model)

    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)
    _sequential_epoch_end(modifier, state, model)
    modifier.end_calibration(model)

    assert model[0].quantization_status == QuantizationStatus.FROZEN
