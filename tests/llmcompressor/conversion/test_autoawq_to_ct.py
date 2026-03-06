"""Tests for the AutoAWQ → compressed-tensors conversion tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from llmcompressor.conversion.autoawq_to_ct import (
    _AWQ_ORDER,
    _AWQ_REVERSE_ORDER,
    _pack_ct_int4,
    _rename_key,
    _repack_awq_to_ct,
    _unpack_awq_int4,
    convert_autoawq_to_ct,
)


# ---------------------------------------------------------------------------
# Helpers – emulate AutoAWQ packing
# ---------------------------------------------------------------------------


def _pack_awq_reference(values: torch.Tensor) -> torch.Tensor:
    """Pack int4 values using the *exact* AutoAWQ interleaved order.

    Reference implementation taken from the AutoAWQ ``gemm_pack`` source.
    """
    rows, cols = values.shape
    assert cols % 8 == 0
    packed = torch.zeros((rows, cols // 8), dtype=torch.int32)
    for col in range(cols // 8):
        for i in range(8):
            packed[:, col] |= (
                (values[:, col * 8 + _AWQ_ORDER[i]] & 0xF).to(torch.int32) << (i * 4)
            )
    return packed


# ---------------------------------------------------------------------------
# Unit: packing round-trip
# ---------------------------------------------------------------------------


class TestUnpackAwqInt4:
    """Verify that AWQ → unpacked → CT-packed gives the right result."""

    def test_roundtrip_identity(self):
        """Pack with AWQ order, unpack, and compare to original."""
        torch.manual_seed(42)
        original = torch.randint(0, 16, (4, 64), dtype=torch.int32)
        packed = _pack_awq_reference(original)
        unpacked = _unpack_awq_int4(packed)
        torch.testing.assert_close(unpacked, original)

    def test_single_group(self):
        """Verify a hand-crafted single group of 8 values."""
        #              idx: 0  1  2  3  4  5  6  7
        vals = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        packed = _pack_awq_reference(vals)
        recovered = _unpack_awq_int4(packed)
        torch.testing.assert_close(recovered, vals)

    def test_max_values(self):
        """All-15 (0xF) should survive the round-trip."""
        original = torch.full((2, 16), 15, dtype=torch.int32)
        packed = _pack_awq_reference(original)
        unpacked = _unpack_awq_int4(packed)
        torch.testing.assert_close(unpacked, original)


class TestPackCtInt4:
    """Verify sequential (compressed-tensors) packing."""

    def test_sequential_pack(self):
        vals = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        packed = _pack_ct_int4(vals)
        # Manually verify: val[i] at bits [4i .. 4i+3]
        expected = 0
        for i in range(8):
            expected |= i << (4 * i)
        assert packed.item() == expected

    def test_columns_not_divisible_by_8(self):
        with pytest.raises(ValueError, match="divisible by 8"):
            _pack_ct_int4(torch.zeros((2, 5), dtype=torch.int32))


class TestRepackAwqToCt:
    """End-to-end: AWQ packed → CT packed."""

    def test_repack(self):
        torch.manual_seed(0)
        original = torch.randint(0, 16, (8, 32), dtype=torch.int32)
        awq_packed = _pack_awq_reference(original)
        ct_packed = _repack_awq_to_ct(awq_packed)
        # Verify by unpacking CT format sequentially
        ct_unpacked = torch.zeros_like(original)
        for i in range(8):
            ct_unpacked[:, i::8] = (ct_packed >> (i * 4)) & 0xF
        torch.testing.assert_close(ct_unpacked, original)


# ---------------------------------------------------------------------------
# Unit: key renaming
# ---------------------------------------------------------------------------


class TestRenameKey:
    def test_qweight(self):
        prefixes = {"model.layers.0.self_attn.q_proj"}
        assert (
            _rename_key(
                "model.layers.0.self_attn.q_proj.qweight", prefixes
            )
            == "model.layers.0.self_attn.q_proj.weight_packed"
        )

    def test_scales(self):
        prefixes = {"model.layers.0.mlp.gate_proj"}
        assert (
            _rename_key("model.layers.0.mlp.gate_proj.scales", prefixes)
            == "model.layers.0.mlp.gate_proj.weight_scale"
        )

    def test_qzeros(self):
        prefixes = {"model.layers.0.mlp.up_proj"}
        assert (
            _rename_key("model.layers.0.mlp.up_proj.qzeros", prefixes)
            == "model.layers.0.mlp.up_proj.weight_zero_point"
        )

    def test_passthrough(self):
        prefixes = {"model.layers.0.self_attn.q_proj"}
        key = "model.embed_tokens.weight"
        assert _rename_key(key, prefixes) == key


# ---------------------------------------------------------------------------
# Inverse permutation sanity
# ---------------------------------------------------------------------------


def test_awq_order_is_valid_permutation():
    assert sorted(_AWQ_ORDER) == list(range(8))


def test_reverse_is_inverse():
    """_AWQ_REVERSE_ORDER must be the algebraic inverse of _AWQ_ORDER."""
    inverse = [0] * 8
    for i, v in enumerate(_AWQ_ORDER):
        inverse[v] = i
    assert _AWQ_REVERSE_ORDER == inverse


# ---------------------------------------------------------------------------
# Integration: full conversion with fake model
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_awq_model(tmp_path: Path) -> Path:
    """Create a minimal fake AutoAWQ model directory with one linear layer."""
    model_dir = tmp_path / "awq_model"
    model_dir.mkdir()

    # Create fake quantised tensors
    torch.manual_seed(123)
    out_features, in_features = 16, 64
    group_size = 16
    n_groups = in_features // group_size

    # Original int4 weights (ground truth)
    original_weights = torch.randint(0, 16, (out_features, in_features), dtype=torch.int32)
    original_zeros = torch.randint(0, 16, (n_groups, out_features), dtype=torch.int32)

    # Pack with AWQ interleaved order
    qweight = _pack_awq_reference(original_weights)
    qzeros = _pack_awq_reference(original_zeros)
    scales = torch.randn(n_groups, out_features, dtype=torch.float16)

    # Non-quantised embedding
    embed = torch.randn(32, 16, dtype=torch.float16)

    tensors = {
        "model.layers.0.self_attn.q_proj.qweight": qweight,
        "model.layers.0.self_attn.q_proj.scales": scales,
        "model.layers.0.self_attn.q_proj.qzeros": qzeros,
        "model.embed_tokens.weight": embed,
    }
    save_file(tensors, str(model_dir / "model.safetensors"))

    # Minimal config.json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "vocab_size": 32,
        "quantization_config": {
            "bits": 4,
            "group_size": group_size,
            "quant_method": "awq",
            "zero_point": True,
        },
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Stash ground-truth for assertions
    torch.save(
        {"weights": original_weights, "zeros": original_zeros},
        model_dir / "_ground_truth.pt",
    )

    return model_dir


def test_convert_autoawq_to_ct(fake_awq_model: Path, tmp_path: Path):
    """Full conversion pipeline: verify tensor contents and config."""
    output_dir = tmp_path / "ct_model"
    convert_autoawq_to_ct(model_path=fake_awq_model, output_path=output_dir)

    # --- config.json ---
    with open(output_dir / "config.json") as f:
        cfg = json.load(f)
    qcfg = cfg["quantization_config"]
    assert qcfg["quant_method"] == "compressed-tensors"
    assert qcfg["format"] == "pack_quantized"
    group_cfg = qcfg["config_groups"]["group_0"]["weights"]
    assert group_cfg["num_bits"] == 4
    assert group_cfg["group_size"] == 16
    assert group_cfg["symmetric"] is False

    # --- safetensors ---
    from safetensors import safe_open

    with safe_open(str(output_dir / "model.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        assert "model.layers.0.self_attn.q_proj.weight_packed" in keys
        assert "model.layers.0.self_attn.q_proj.weight_scale" in keys
        assert "model.layers.0.self_attn.q_proj.weight_zero_point" in keys
        assert "model.embed_tokens.weight" in keys

        # Old AWQ keys must be gone
        assert "model.layers.0.self_attn.q_proj.qweight" not in keys
        assert "model.layers.0.self_attn.q_proj.scales" not in keys
        assert "model.layers.0.self_attn.q_proj.qzeros" not in keys

        # Verify weight values are correct after repacking
        ct_packed = f.get_tensor(
            "model.layers.0.self_attn.q_proj.weight_packed"
        )
        ground_truth = torch.load(
            fake_awq_model / "_ground_truth.pt", weights_only=True
        )

        # Unpack CT format (sequential) and compare to ground truth
        rows, packed_cols = ct_packed.shape
        ct_unpacked = torch.zeros(
            (rows, packed_cols * 8), dtype=torch.int32
        )
        for i in range(8):
            ct_unpacked[:, i::8] = (ct_packed >> (i * 4)) & 0xF

        torch.testing.assert_close(ct_unpacked, ground_truth["weights"])
