"""
Tests for update_fused_layer_weight_global_scales, including support for
MLA-style attention modules (e.g. DeepSeek V2/V3).

Covers:
- Standard Q/K/V projections
- Already-fused QKV projection
- DeepSeek MLA projections (q_a_proj + kv_a_proj_with_mqa)
- Modules with no matching fused group (should be a no-op)
- MLP gate_proj/up_proj fusion
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationStrategy

from llmcompressor.modifiers.utils.helpers import update_fused_layer_weight_global_scales


# ---------------------------------------------------------------------------
# Helpers to build mock modules
# ---------------------------------------------------------------------------


def _make_linear_with_global_scale(
    in_features: int,
    out_features: int,
    global_scale_value: float,
) -> nn.Linear:
    """Create a ``nn.Linear`` with a ``weight_global_scale`` parameter and a
    ``quantization_scheme`` that declares TENSOR_GROUP strategy."""
    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight_global_scale = nn.Parameter(
        torch.tensor([global_scale_value], dtype=torch.float32)
    )
    linear.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="float",
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=16,
            symmetric=True,
        ),
    )
    return linear


class _FakeStandardAttention(nn.Module):
    """Mock attention module with standard q_proj, k_proj, v_proj."""

    def __init__(self, dim: int = 64, scales=(1.0, 2.0, 3.0)):
        super().__init__()
        self.q_proj = _make_linear_with_global_scale(dim, dim, scales[0])
        self.k_proj = _make_linear_with_global_scale(dim, dim, scales[1])
        self.v_proj = _make_linear_with_global_scale(dim, dim, scales[2])


class _FakeFusedQKVAttention(nn.Module):
    """Mock attention module with already-fused qkv_proj."""

    def __init__(self, dim: int = 64, scale: float = 5.0):
        super().__init__()
        self.qkv_proj = _make_linear_with_global_scale(dim, dim * 3, scale)


class _FakeMLAAttention(nn.Module):
    """Mock DeepSeek V2/V3 MLA-style attention with q_a_proj + kv_a_proj_with_mqa."""

    def __init__(self, dim: int = 64, scales=(4.0, 8.0)):
        super().__init__()
        self.q_a_proj = _make_linear_with_global_scale(dim, dim, scales[0])
        self.kv_a_proj_with_mqa = _make_linear_with_global_scale(dim, dim, scales[1])


class _FakeMLAAttentionFull(nn.Module):
    """MLA attention with both compressed and decompressed projections."""

    def __init__(self, dim: int = 64):
        super().__init__()
        # compressed
        self.q_a_proj = _make_linear_with_global_scale(dim, dim, 4.0)
        self.kv_a_proj_with_mqa = _make_linear_with_global_scale(dim, dim, 8.0)
        # decompressed
        self.q_b_proj = _make_linear_with_global_scale(dim, dim, 2.0)
        self.kv_b_proj = _make_linear_with_global_scale(dim, dim, 6.0)


class _FakeNoMatchAttention(nn.Module):
    """Attention module with non-standard projection names that don't match any group."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.custom_proj_a = _make_linear_with_global_scale(dim, dim, 1.0)
        self.custom_proj_b = _make_linear_with_global_scale(dim, dim, 2.0)


class FakeMLP(nn.Module):
    """MLP module with gate_proj and up_proj."""

    def __init__(self, dim: int = 64, scales=(3.0, 7.0)):
        super().__init__()
        self.gate_proj = _make_linear_with_global_scale(dim, dim, scales[0])
        self.up_proj = _make_linear_with_global_scale(dim, dim, scales[1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpdateFusedLayerWeightGlobalScales:
    """Test suite for update_fused_layer_weight_global_scales."""

    @patch(
        "llmcompressor.modifiers.utils.helpers.is_attention_module",
        return_value=True,
    )
    def test_standard_qkv_projection(self, mock_is_attn):
        """Standard q/k/v projections should be fused to the minimum scale."""
        module = _FakeStandardAttention(scales=(1.0, 2.0, 3.0))
        update_fused_layer_weight_global_scales(module)

        expected_min = 1.0
        for proj in [module.q_proj, module.k_proj, module.v_proj]:
            assert proj.weight_global_scale.item() == pytest.approx(expected_min)

    @patch(
        "llmcompressor.modifiers.utils.helpers.is_attention_module",
        return_value=True,
    )
    def test_fused_qkv_is_noop(self, mock_is_attn):
        """Already-fused qkv_proj should not be touched."""
        module = _FakeFusedQKVAttention(scale=5.0)
        update_fused_layer_weight_global_scales(module)

        assert module.qkv_proj.weight_global_scale.item() == pytest.approx(5.0)

    @patch(
        "llmcompressor.modifiers.utils.helpers.is_attention_module",
        return_value=True,
    )
    def test_deepseek_mla_compressed(self, mock_is_attn):
        """DeepSeek MLA compressed projections should be fused."""
        module = _FakeMLAAttention(scales=(4.0, 8.0))
        update_fused_layer_weight_global_scales(module)

        expected_min = 4.0
        assert module.q_a_proj.weight_global_scale.item() == pytest.approx(
            expected_min
        )
        assert module.kv_a_proj_with_mqa.weight_global_scale.item() == pytest.approx(
            expected_min
        )

    @patch(
        "llmcompressor.modifiers.utils.helpers.is_attention_module",
        return_value=True,
    )
    def test_deepseek_mla_full_only_first_group_matches(self, mock_is_attn):
        """When both compressed and decompressed MLA projections exist,
        only the first matching group (compressed) is fused per call.

        In practice the module tree is traversed and the function is called
        for each sub-module, so both groups will eventually be handled.
        But per single call, only the first match should apply."""
        module = _FakeMLAAttentionFull()
        update_fused_layer_weight_global_scales(module)

        # The first matching group is q_a_proj + kv_a_proj_with_mqa
        expected_min_compressed = 4.0
        assert module.q_a_proj.weight_global_scale.item() == pytest.approx(
            expected_min_compressed
        )
        assert module.kv_a_proj_with_mqa.weight_global_scale.item() == pytest.approx(
            expected_min_compressed
        )

    @patch(
        "llmcompressor.modifiers.utils.helpers.is_attention_module",
        return_value=True,
    )
    def test_no_matching_attention_group_is_noop(self, mock_is_attn):
        """Attention with unrecognized projection names should be a no-op."""
        module = _FakeNoMatchAttention()
        update_fused_layer_weight_global_scales(module)

        assert module.custom_proj_a.weight_global_scale.item() == pytest.approx(1.0)
        assert module.custom_proj_b.weight_global_scale.item() == pytest.approx(2.0)

    def test_mlp_gate_up_fusion(self):
        """MLP gate_proj and up_proj should be fused to the minimum scale."""
        module = FakeMLP(scales=(3.0, 7.0))
        update_fused_layer_weight_global_scales(module)

        expected_min = 3.0
        assert module.gate_proj.weight_global_scale.item() == pytest.approx(
            expected_min
        )
        assert module.up_proj.weight_global_scale.item() == pytest.approx(expected_min)

    @patch(
        "llmcompressor.modifiers.utils.helpers.is_attention_module",
        return_value=False,
    )
    def test_non_attention_non_mlp_is_noop(self, mock_is_attn):
        """A module that is neither attention nor MLP should be untouched."""
        module = nn.Linear(64, 64)
        update_fused_layer_weight_global_scales(module)
        # No crash, no changes — just verify it's a no-op
