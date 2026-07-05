"""
Test that start_calibration initializes KV cache observers on attention modules
regardless of their name in the module tree.

compressed_tensors' _apply_kv_cache_scheme discovers attention modules via
is_attention_module() (name-agnostic), but QuantizationMixin.start_calibration
previously relied on resolved_targets which includes "re:.*self_attn$". This
regex fails for models that name their attention differently (e.g. "attention",
"self_attention"). The fix adds an is_attention_module() fallback pass.
"""

import torch.nn as nn
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStatus,
    apply_quantization_config,
    is_attention_module,
)
from transformers import PretrainedConfig

from llmcompressor.modifiers.quantization.quantization import QuantizationModifier


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
        # Named "attention" instead of "self_attn" — does NOT match
        # the "re:.*self_attn$" regex in resolved_targets.
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


def test_attention_module_not_named_self_attn_gets_calibrated():
    """Attention modules named 'attention' (not 'self_attn') must still
    get observers and hooks initialized when kv_cache_scheme is set."""
    model = _StubModel(dim=16)
    modifier = QuantizationModifier(
        targets=["Linear"],
        kv_cache_scheme=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            dynamic=False,
            symmetric=True,
        ),
    )

    # Verify our stub is recognized as attention
    attn_modules = [
        (name, m) for name, m in model.named_modules() if is_attention_module(m)
    ]
    assert len(attn_modules) == 2, "Expected 2 attention modules in _StubModel"

    # Verify none of them match the self_attn regex
    for name, _ in attn_modules:
        assert "self_attn" not in name, f"{name} unexpectedly matches self_attn"

    # Apply quantization config (this uses is_attention_module and WILL set scheme)
    apply_quantization_config(model, modifier.resolved_config)

    # Verify schemes were applied to attention modules by _apply_kv_cache_scheme
    for _, m in attn_modules:
        assert hasattr(m, "quantization_scheme"), (
            "apply_quantization_config should set "
            "quantization_scheme on attention modules"
        )

    # Now run start_calibration — this is what we're testing
    modifier.start_calibration(model)

    # Verify attention modules got calibration status
    for name, m in attn_modules:
        assert m.quantization_status == QuantizationStatus.CALIBRATION, (
            f"Attention module '{name}' was not calibrated — "
            f"start_calibration missed it (status={m.quantization_status})"
        )

    # Verify hooks were registered for KV cache calibration
    assert (
        len(modifier._calibration_hooks) > 0
    ), "Expected calibration hooks to be registered"

    # Clean up
    modifier.end_calibration(model)

    for name, m in attn_modules:
        assert (
            m.quantization_status == QuantizationStatus.FROZEN
        ), f"Attention module '{name}' was not frozen after end_calibration"
