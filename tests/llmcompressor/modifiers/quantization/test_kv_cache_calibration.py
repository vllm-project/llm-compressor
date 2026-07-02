"""
Tests for calibration validation on quantized modules.

KV cache quantization relies on key/value observers collecting calibration
statistics before the final k_scale/v_scale values are frozen. These tests use
small CPU-only stubs so the calibration lifecycle can be validated without
downloading a model.
"""

import pytest
import torch
import torch.nn as nn
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
    is_attention_module,
)
from transformers import PretrainedConfig

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.quantization.calibration import observe, update_qparams
from llmcompressor.modifiers.quantization.quantization import QuantizationModifier
from llmcompressor.observers.min_max import StaticMinMaxObserver


class _StubAttention(nn.Module):
    """Minimal attention module recognized by is_attention_module().

    is_attention_module checks: "attention" in class name (lowercase) and
    hasattr(k_proj) or hasattr(v_proj).
    """

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


class _EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)

    def forward(self, input_ids):
        return self.embed(input_ids)


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


def _linear_modifier():
    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    return QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=args,
                input_activations=args,
            )
        }
    )


def _embedding_modifier():
    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    return QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(targets=["Embedding"], weights=args)
        }
    )


def _attention_modules(model):
    return [(name, m) for name, m in model.named_modules() if is_attention_module(m)]


def _prepare_model(
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


def test_observer_tracks_num_observations():
    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    observer = StaticMinMaxObserver(base_name="input", args=args)

    assert observer.num_observations == 0
    assert not observer.has_statistics

    observer(torch.empty(0))
    assert observer.num_observations == 0

    observer(torch.ones(2, 4))
    assert observer.num_observations == 1
    assert observer.has_statistics

    observer(torch.full((2, 4), 2.0))
    assert observer.num_observations == 2


def test_attention_module_not_named_self_attn_gets_calibrated():
    """Attention modules named 'attention' should still get KV observers/hooks."""
    model, modifier, attn_modules = _prepare_model(dim=16)

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


def test_kv_cache_calibration_requires_observed_scales():
    model, modifier, attn_modules = _prepare_model(dim=16, num_layers=1)
    _observe_weight_observers(model)

    with pytest.raises(ValueError) as exc_info:
        modifier.end_calibration(model)

    error_message = str(exc_info.value)
    assert "Quantization calibration failed" in error_message
    assert "layers.0.attention.k_observer" in error_message
    assert "layers.0.attention.v_observer" in error_message
    assert "invalid static KV-cache scales" not in error_message

    attention = attn_modules[0][1]
    assert attention.quantization_status == QuantizationStatus.CALIBRATION
    assert hasattr(attention, "k_observer")
    assert hasattr(attention, "v_observer")


def test_kv_cache_calibration_rejects_zero_and_non_finite_scales():
    model, modifier, attn_modules = _prepare_model(dim=16, num_layers=1)
    _observe_weight_observers(model)
    _observe_kv_cache(attn_modules)
    attn_modules[0][1].k_scale.data.zero_()
    attn_modules[0][1].v_scale.data.fill_(float("nan"))

    with pytest.raises(ValueError) as exc_info:
        modifier.end_calibration(model)

    error_message = str(exc_info.value)
    assert "layers.0.attention.k_scale" in error_message
    assert "non-positive" in error_message
    assert "layers.0.attention.v_scale" in error_message
    assert "non-finite" in error_message

    attention = attn_modules[0][1]
    assert attention.quantization_status == QuantizationStatus.CALIBRATION
    assert hasattr(attention, "k_observer")
    assert hasattr(attention, "v_observer")


def test_kv_cache_calibration_respects_ignore_list():
    model, modifier, attn_modules = _prepare_model(
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
    model, modifier, _ = _prepare_model(dim=16, num_layers=1, dynamic=True)
    _observe_weight_observers(model)

    modifier.end_calibration(model)


def test_local_dynamic_kv_cache_calibration_skips_static_scale_validation():
    model, modifier, _ = _prepare_model(dim=16, num_layers=1, dynamic="local")
    _observe_weight_observers(model)

    modifier.end_calibration(model)


def test_end_calibration_can_run_after_kv_cache_is_frozen():
    model, modifier, attn_modules = _prepare_model(dim=16, num_layers=1)
    _observe_weight_observers(model)
    _observe_kv_cache(attn_modules)

    modifier.end_calibration(model)
    attn_modules[0][1].k_scale.data.zero_()
    modifier.end_calibration(model)

    assert attn_modules[0][1].quantization_status == QuantizationStatus.FROZEN


def test_kv_cache_validation_skips_modules_without_quantization_scheme():
    model, modifier, attn_modules = _prepare_model(dim=16, num_layers=1)
    delattr(attn_modules[0][1], "quantization_scheme")
    _observe_weight_observers(model)

    modifier.end_calibration(model)


def test_sequential_epoch_end_validates_current_linear_chunk():
    model = nn.Sequential(nn.Linear(4, 4))
    modifier = _linear_modifier()
    state = State(model=model)

    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)

    with pytest.raises(ValueError) as exc_info:
        modifier.on_sequential_epoch_end(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_END),
            modules=list(model.modules()),
        )

    error_message = str(exc_info.value)
    assert "0.input_observer" in error_message
    assert "0.weight_observer" not in error_message


def test_embedding_weight_observer_is_validated():
    model = _EmbeddingModel()
    modifier = _embedding_modifier()
    state = State(model=model)

    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)

    with pytest.raises(ValueError) as exc_info:
        modifier.validate_module_calibration(model)

    assert "embed.weight_observer" in str(exc_info.value)

    observe(model.embed, "weight")
    update_qparams(model.embed, "weight")
    modifier.validate_module_calibration(model)
