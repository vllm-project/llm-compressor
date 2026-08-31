"""
Tests for MTP (Multi-Token Prediction) layer quantization helpers in
compressed_tensors_utils.  All tests are self-contained — they build local
dummy checkpoints rather than downloading real models, so no HF hub access or
GPU is required.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from llmcompressor.transformers.compression.compressed_tensors_utils import (
    _extract_mtp_scheme,
    _get_mtp_prefix,
    _quantize_and_save_mtp_tensors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source_checkpoint(tmp_dir, tensors: dict, prefix_style: str = "mtp"):
    """Write a minimal safetensors checkpoint with MTP tensors."""
    if prefix_style == "mtp":
        mtp_tensors = {
            "mtp.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
            "mtp.layers.0.self_attn.v_proj.weight": torch.randn(32, 32),
            "mtp.layers.0.mlp.gate_proj.weight": torch.randn(64, 32),
            "mtp.norm.weight": torch.randn(32),
        }
        all_tensors = {**tensors, **mtp_tensors}
        # multi-shard layout: main shard + mtp shard + index
        save_file(tensors, os.path.join(tmp_dir, "model.safetensors"))
        save_file(mtp_tensors, os.path.join(tmp_dir, "model_mtp.safetensors"))
        weight_map = {k: "model.safetensors" for k in tensors}
        weight_map.update({k: "model_mtp.safetensors" for k in mtp_tensors})
        index = {"metadata": {}, "weight_map": weight_map}
        with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
    elif prefix_style == "glm":
        # GLM-style: MTP at model.layers.{num_hidden}
        num_hidden = 4
        mtp_tensors = {
            f"model.layers.{num_hidden}.self_attn.q_proj.weight": torch.randn(32, 32),
            f"model.layers.{num_hidden}.mlp.gate_proj.weight": torch.randn(64, 32),
        }
        all_tensors = {**tensors, **mtp_tensors}
        save_file(all_tensors, os.path.join(tmp_dir, "model.safetensors"))
        weight_map = {k: "model.safetensors" for k in all_tensors}
        index = {"metadata": {}, "weight_map": weight_map}
        with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
    return tmp_dir


def _make_dest_checkpoint(tmp_dir, quant_config: dict | None = None):
    """Write a minimal dest checkpoint with config.json and a main shard."""
    save_file(
        {"model.embed_tokens.weight": torch.zeros(10, 8)},
        os.path.join(tmp_dir, "model.safetensors"),
    )
    cfg = {"model_type": "test"}
    if quant_config is not None:
        cfg["quantization_config"] = quant_config
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    return tmp_dir


# ---------------------------------------------------------------------------
# _extract_mtp_scheme
# ---------------------------------------------------------------------------


def test_extract_mtp_scheme_fp8_tensor():
    scheme = _extract_mtp_scheme(
        {
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 8, "type": "float", "strategy": "tensor"},
                }
            },
            "format": "dense",
        }
    )
    assert scheme is not None
    assert scheme.weights.num_bits == 8
    assert str(scheme.weights.type) in ("float", "QuantizationType.float")
    assert str(scheme.weights.strategy) in ("tensor", "QuantizationStrategy.tensor")


def test_extract_mtp_scheme_w4a16():
    scheme = _extract_mtp_scheme(
        {
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "group",
                        "group_size": 128,
                    },
                }
            },
            "format": "dense",
        }
    )
    assert scheme is not None
    assert scheme.weights.num_bits == 4
    assert str(scheme.weights.type) in ("int", "QuantizationType.int")


def test_extract_mtp_scheme_nvfp4_falls_back_to_fp8_block():
    scheme = _extract_mtp_scheme(
        {
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "type": "float", "strategy": "tensor"},
                }
            },
            "format": "nvfp4-pack-quantized",
        }
    )
    assert scheme is not None
    # should fall back to FP8-block
    assert scheme.weights.num_bits == 8
    assert str(scheme.weights.strategy) in ("block", "QuantizationStrategy.block")


def test_extract_mtp_scheme_returns_none_for_empty():
    assert _extract_mtp_scheme({}) is None
    assert _extract_mtp_scheme({"config_groups": {}}) is None
    assert (
        _extract_mtp_scheme({"config_groups": {"g": {"targets": ["Linear"]}}}) is None
    )


# ---------------------------------------------------------------------------
# _get_mtp_prefix
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, num_hidden_layers=4):
        self.num_hidden_layers = num_hidden_layers


def test_get_mtp_prefix_standard_mtp(tmp_path):
    main = {"model.embed_tokens.weight": torch.zeros(4, 4)}
    _make_source_checkpoint(str(tmp_path), main, prefix_style="mtp")
    prefix = _get_mtp_prefix(str(tmp_path), _FakeConfig())
    assert prefix == "mtp"


def test_get_mtp_prefix_glm_style(tmp_path):
    main = {
        f"model.layers.{i}.mlp.gate_proj.weight": torch.zeros(4, 4) for i in range(4)
    }
    _make_source_checkpoint(str(tmp_path), main, prefix_style="glm")
    prefix = _get_mtp_prefix(str(tmp_path), _FakeConfig(num_hidden_layers=4))
    assert prefix == "model.layers.4"


def test_get_mtp_prefix_raises_when_undetectable(tmp_path):
    tensors = {"model.layers.0.weight": torch.zeros(4, 4)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    weight_map = {k: "model.safetensors" for k in tensors}
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    with pytest.raises(ValueError, match="Could not detect MTP tensor prefix"):
        _get_mtp_prefix(str(tmp_path), _FakeConfig(num_hidden_layers=99))


# ---------------------------------------------------------------------------
# _quantize_and_save_mtp_tensors
# ---------------------------------------------------------------------------


FP8_QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "float", "strategy": "tensor"},
        }
    },
    "format": "dense",
    "quantization_status": "compressed",
}


def test_quantize_and_save_mtp_creates_shard(tmp_path):
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    main = {"model.layers.0.weight": torch.randn(32, 32)}
    _make_source_checkpoint(src, main, prefix_style="mtp")
    _make_dest_checkpoint(dst, quant_config=FP8_QUANT_CONFIG)

    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp")

    assert os.path.exists(os.path.join(dst, "model_mtp.safetensors"))
    assert os.path.exists(os.path.join(dst, "model.safetensors.index.json"))


def test_quantize_and_save_mtp_index_includes_mtp_keys(tmp_path):
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    main = {"model.layers.0.weight": torch.randn(32, 32)}
    _make_source_checkpoint(src, main, prefix_style="mtp")
    _make_dest_checkpoint(dst, quant_config=FP8_QUANT_CONFIG)

    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp")

    with open(os.path.join(dst, "model.safetensors.index.json")) as f:
        index = json.load(f)
    mtp_keys = [k for k in index["weight_map"] if k.startswith("mtp")]
    assert len(mtp_keys) > 0


def test_quantize_and_save_mtp_config_group_added(tmp_path):
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    main = {"model.layers.0.weight": torch.randn(32, 32)}
    _make_source_checkpoint(src, main, prefix_style="mtp")
    _make_dest_checkpoint(dst, quant_config=FP8_QUANT_CONFIG)

    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp")

    with open(os.path.join(dst, "config.json")) as f:
        cfg = json.load(f)
    groups = cfg.get("quantization_config", {}).get("config_groups", {})
    assert "mtp_group" in groups
    # MTP should NOT be in the ignore list
    ignore = cfg.get("quantization_config", {}).get("ignore", [])
    assert not any("mtp" in s for s in ignore)


def test_quantize_and_save_mtp_unquantized_fallback_adds_ignore(tmp_path):
    """When no quantization_config is present, MTP tensors are saved unquantized
    and marked as ignored so inference engines skip them."""
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    main = {"model.layers.0.weight": torch.randn(32, 32)}
    _make_source_checkpoint(src, main, prefix_style="mtp")
    # dest has a quantization_config but with no config_groups → scheme is None
    _make_dest_checkpoint(dst, quant_config={"format": "dense"})

    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp")

    with open(os.path.join(dst, "config.json")) as f:
        cfg = json.load(f)
    ignore = cfg.get("quantization_config", {}).get("ignore", [])
    assert any("mtp" in s for s in ignore)
