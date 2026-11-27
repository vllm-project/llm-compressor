"""Tests for cross-layer quantization scheme consistency validation."""

import logging

import pytest
import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.modifiers.quantization.scheme_consistency_validation import (
    get_expert_scheme_mismatches,
    get_fused_group_mismatches,
    validate_scheme_consistency,
)


# ---------------------------------------------------------------------------
# Test model helpers
# ---------------------------------------------------------------------------


class _AttentionBlock(torch.nn.Module):
    """Mimics a transformer attention block with q/k/v projections."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)


class _MLPBlock(torch.nn.Module):
    """Mimics a transformer MLP block with gate/up projections."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.gate_proj = torch.nn.Linear(dim, dim)
        self.up_proj = torch.nn.Linear(dim, dim)
        self.down_proj = torch.nn.Linear(dim, dim)


class _TransformerLayer(torch.nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.self_attn = _AttentionBlock(dim)
        self.mlp = _MLPBlock(dim)


class _Expert(torch.nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.gate_proj = torch.nn.Linear(dim, dim)
        self.up_proj = torch.nn.Linear(dim, dim)
        self.down_proj = torch.nn.Linear(dim, dim)


class _MoELayer(torch.nn.Module):
    def __init__(self, num_experts: int = 4, dim: int = 64):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [_Expert(dim) for _ in range(num_experts)]
        )


class _MoEModel(torch.nn.Module):
    def __init__(self, num_experts: int = 4, dim: int = 64):
        super().__init__()
        self.layer = _MoELayer(num_experts, dim)


def _w4_scheme() -> QuantizationScheme:
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, strategy="group", group_size=128),
    )


def _w8_scheme() -> QuantizationScheme:
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8, strategy="channel"),
    )


def _fp8_scheme() -> QuantizationScheme:
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
    )


def _attach_scheme(module: torch.nn.Module, scheme: QuantizationScheme):
    module.quantization_scheme = scheme


# ---------------------------------------------------------------------------
# Fused group tests
# ---------------------------------------------------------------------------


class TestFusedGroupMismatches:
    def test_no_mismatch_when_all_match(self):
        model = _TransformerLayer()
        scheme = _w4_scheme()
        for proj in (model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj):
            _attach_scheme(proj, scheme)
        for proj in (model.mlp.gate_proj, model.mlp.up_proj):
            _attach_scheme(proj, scheme)

        assert get_fused_group_mismatches(model) == []

    def test_qkv_mismatch_detected(self):
        model = _TransformerLayer()
        _attach_scheme(model.self_attn.q_proj, _w4_scheme())
        _attach_scheme(model.self_attn.k_proj, _w8_scheme())
        _attach_scheme(model.self_attn.v_proj, _w4_scheme())

        mismatches = get_fused_group_mismatches(model)
        assert len(mismatches) >= 1

        parent_fqn, group_names, sigs = mismatches[0]
        assert "q_proj" in group_names and "k_proj" in group_names
        assert sigs["q_proj"] != sigs["k_proj"]

    def test_gate_up_mismatch_detected(self):
        model = _TransformerLayer()
        _attach_scheme(model.mlp.gate_proj, _w4_scheme())
        _attach_scheme(model.mlp.up_proj, _fp8_scheme())

        mismatches = get_fused_group_mismatches(model)
        gate_up_mismatches = [
            m for m in mismatches if "gate_proj" in m[1]
        ]
        assert len(gate_up_mismatches) == 1
        assert gate_up_mismatches[0][2]["gate_proj"] != gate_up_mismatches[0][2]["up_proj"]

    def test_no_scheme_attached_skipped(self):
        """Layers without quantization_scheme are silently skipped."""
        model = _TransformerLayer()
        # only attach to q_proj, not k/v
        _attach_scheme(model.self_attn.q_proj, _w4_scheme())

        assert get_fused_group_mismatches(model) == []

    def test_single_layer_with_scheme_skipped(self):
        """A fused group with only one quantized member is not a mismatch."""
        model = _TransformerLayer()
        _attach_scheme(model.self_attn.q_proj, _w4_scheme())

        assert get_fused_group_mismatches(model) == []


# ---------------------------------------------------------------------------
# Expert scheme tests
# ---------------------------------------------------------------------------


class TestExpertSchemeMismatches:
    def test_no_mismatch_when_all_experts_match(self):
        model = _MoEModel(num_experts=4)
        scheme = _w4_scheme()
        for expert in model.layer.experts:
            for child in (expert.gate_proj, expert.up_proj, expert.down_proj):
                _attach_scheme(child, scheme)

        assert get_expert_scheme_mismatches(model) == []

    def test_expert_mismatch_detected(self):
        model = _MoEModel(num_experts=3)
        w4 = _w4_scheme()
        w8 = _w8_scheme()

        # expert 0 and 2 get w4, expert 1 gets w8 on gate_proj
        for expert in model.layer.experts:
            _attach_scheme(expert.up_proj, w4)
            _attach_scheme(expert.down_proj, w4)

        _attach_scheme(model.layer.experts[0].gate_proj, w4)
        _attach_scheme(model.layer.experts[1].gate_proj, w8)
        _attach_scheme(model.layer.experts[2].gate_proj, w4)

        mismatches = get_expert_scheme_mismatches(model)
        assert len(mismatches) >= 1

        experts_fqn, sublayer_name, sigs = mismatches[0]
        assert sublayer_name == "gate_proj"
        assert sigs["0"] != sigs["1"]

    def test_no_scheme_on_experts_skipped(self):
        model = _MoEModel(num_experts=2)
        # no schemes attached
        assert get_expert_scheme_mismatches(model) == []


# ---------------------------------------------------------------------------
# Integration: validate_scheme_consistency
# ---------------------------------------------------------------------------


class TestValidateSchemeConsistency:
    def test_no_warning_when_consistent(self, caplog):
        model = _TransformerLayer()
        scheme = _w4_scheme()
        for proj in (model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj):
            _attach_scheme(proj, scheme)
        for proj in (model.mlp.gate_proj, model.mlp.up_proj):
            _attach_scheme(proj, scheme)

        with caplog.at_level(logging.WARNING):
            validate_scheme_consistency(model)
        assert "Inconsistent quantization schemes" not in caplog.text

    def test_warning_on_fused_mismatch(self, caplog):
        model = _TransformerLayer()
        _attach_scheme(model.self_attn.q_proj, _w4_scheme())
        _attach_scheme(model.self_attn.k_proj, _w8_scheme())
        _attach_scheme(model.self_attn.v_proj, _w4_scheme())

        with caplog.at_level(logging.WARNING):
            validate_scheme_consistency(model)
        assert "Inconsistent quantization schemes" in caplog.text
        assert "q_proj" in caplog.text
        assert "k_proj" in caplog.text

    def test_warning_on_expert_mismatch(self, caplog):
        model = _MoEModel(num_experts=2)
        _attach_scheme(model.layer.experts[0].gate_proj, _w4_scheme())
        _attach_scheme(model.layer.experts[1].gate_proj, _w8_scheme())

        with caplog.at_level(logging.WARNING):
            validate_scheme_consistency(model)
        assert "Inconsistent quantization schemes" in caplog.text
        assert "expert" in caplog.text.lower()

    def test_combined_fused_and_expert_mismatch(self, caplog):
        """Both fused and expert mismatches are reported together."""

        class _CombinedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _AttentionBlock()
                self.moe = _MoELayer(num_experts=2)

        model = _CombinedModel()
        _attach_scheme(model.self_attn.q_proj, _w4_scheme())
        _attach_scheme(model.self_attn.k_proj, _w8_scheme())
        _attach_scheme(model.self_attn.v_proj, _w4_scheme())
        _attach_scheme(model.moe.experts[0].gate_proj, _w4_scheme())
        _attach_scheme(model.moe.experts[1].gate_proj, _fp8_scheme())

        with caplog.at_level(logging.WARNING):
            validate_scheme_consistency(model)
        assert "Fused layer group" in caplog.text
        assert "Expert sub-layer" in caplog.text
