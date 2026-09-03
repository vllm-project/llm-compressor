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
    _get_mtp_prefix,
    _quantize_and_save_mtp_tensors,
    _resolve_mtp_scheme,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source_checkpoint(tmp_dir, tensors: dict):
    """Write a minimal safetensors checkpoint with MTP tensors."""
    mtp_tensors = {
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        "mtp.layers.0.self_attn.v_proj.weight": torch.randn(32, 32),
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(64, 32),
        "mtp.eh_proj.weight": torch.randn(32, 64),
        "mtp.norm.weight": torch.randn(32),
    }
    # multi-shard layout: main shard + mtp shard + index
    save_file(tensors, os.path.join(tmp_dir, "model.safetensors"))
    save_file(mtp_tensors, os.path.join(tmp_dir, "model_mtp.safetensors"))
    weight_map = {k: "model.safetensors" for k in tensors}
    weight_map.update({k: "model_mtp.safetensors" for k in mtp_tensors})
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
# _get_mtp_prefix
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, num_hidden_layers=4):
        self.num_hidden_layers = num_hidden_layers


def test_get_mtp_prefix_standard_mtp(tmp_path):
    main = {"model.embed_tokens.weight": torch.zeros(4, 4)}
    _make_source_checkpoint(str(tmp_path), main)
    prefix = _get_mtp_prefix(str(tmp_path), _FakeConfig())
    assert prefix == "mtp"


def test_get_mtp_prefix_glm_vlm_style(tmp_path):
    """GLM-5.3-Flash stores MTP at model.language_model.layers.{num_hidden}.*"""
    num_hidden = 4
    tensors = {
        f"model.language_model.layers.{i}.mlp.gate_proj.weight": torch.zeros(4, 4)
        for i in range(num_hidden)
    }
    tensors[f"model.language_model.layers.{num_hidden}.eh_proj.weight"] = torch.zeros(
        4, 4
    )
    save_file(tensors, str(tmp_path / "model.safetensors"))
    weight_map = {k: "model.safetensors" for k in tensors}
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    prefix = _get_mtp_prefix(str(tmp_path), _FakeConfig(num_hidden_layers=num_hidden))
    assert prefix == f"model.language_model.layers.{num_hidden}"


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
    _make_source_checkpoint(src, main)
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
    _make_source_checkpoint(src, main)
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
    _make_source_checkpoint(src, main)
    _make_dest_checkpoint(dst, quant_config=FP8_QUANT_CONFIG)

    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp", mtp_scheme="FP8_DYNAMIC")

    with open(os.path.join(dst, "config.json")) as f:
        cfg = json.load(f)
    groups = cfg.get("quantization_config", {}).get("config_groups", {})
    assert "mtp_group" in groups
    # mtp_group must be first so vLLM's first-match resolution wins for the MTP
    # modules over the main model's broad regexes
    assert next(iter(groups)) == "mtp_group"
    # the group must carry its own compression format (required when the
    # top-level format is mixed-precision, and present on the main groups too)
    assert groups["mtp_group"].get("format") not in (None, "dense")
    targets = groups["mtp_group"]["targets"]
    # targets match module names, not parameter names, so never a ".weight"
    # suffix
    assert all(not t.endswith(".weight") for t in targets)
    # an mtp-anchored regex leads the targets so it matches vLLM's *fused*
    # runtime module names (qkv_proj / gate_up_proj), which the exact
    # per-component names cannot
    assert any(t.startswith("re:") and t[3:].startswith("^mtp") for t in targets)
    # the exact component modules are still listed; the 1D norm is excluded
    assert "mtp.layers.0.self_attn.q_proj" in targets
    assert "mtp.norm" not in targets
    # MTP should NOT be in the ignore list
    ignore = cfg.get("quantization_config", {}).get("ignore", [])
    assert not any("mtp" in s for s in ignore)


def test_quantize_and_save_mtp_keeps_1d_norms_full_precision(tmp_path):
    """1D tensors (e.g. mtp.norm.weight) must stay full precision, not be
    routed through the linear quantization path."""
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    main = {"model.layers.0.weight": torch.randn(32, 32)}
    _make_source_checkpoint(src, main)
    _make_dest_checkpoint(dst, quant_config=FP8_QUANT_CONFIG)

    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp", mtp_scheme="FP8_DYNAMIC")

    from safetensors import safe_open

    with safe_open(os.path.join(dst, "model_mtp.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        # norm stays as a plain weight (no scale companion), unchanged dtype
        assert "mtp.norm.weight" in keys
        assert "mtp.norm.weight_scale" not in keys
        assert f.get_tensor("mtp.norm.weight").dtype == torch.float32
        # a 2D linear weight is quantized (gains a weight_scale)
        assert "mtp.layers.0.self_attn.q_proj.weight_scale" in keys


def test_quantize_and_save_mtp_keeps_embeddings_full_precision(tmp_path):
    """Embedding/head modules must stay full precision: vLLM cannot load FP8
    embeddings, so they are neither quantized nor listed in mtp_group."""
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)

    # source with an embedding (rows == vocab_size) alongside a normal linear
    vocab_size = 128
    mtp_tensors = {
        "mtp.embed_tokens.weight": torch.randn(vocab_size, 32),
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
    }
    save_file(
        {"model.layers.0.weight": torch.randn(32, 32)},
        os.path.join(src, "model.safetensors"),
    )
    save_file(mtp_tensors, os.path.join(src, "model_mtp.safetensors"))
    weight_map = {"model.layers.0.weight": "model.safetensors"}
    weight_map.update({k: "model_mtp.safetensors" for k in mtp_tensors})
    with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    _make_dest_checkpoint(dst, quant_config=FP8_QUANT_CONFIG)

    _quantize_and_save_mtp_tensors(
        src, dst, mtp_prefix="mtp", vocab_size=vocab_size, mtp_scheme="FP8_DYNAMIC"
    )

    from safetensors import safe_open

    with safe_open(os.path.join(dst, "model_mtp.safetensors"), framework="pt") as f:
        keys = set(f.keys())
    assert "mtp.embed_tokens.weight" in keys
    assert "mtp.embed_tokens.weight_scale" not in keys  # not quantized
    assert "mtp.layers.0.self_attn.q_proj.weight_scale" in keys  # linear is

    with open(os.path.join(dst, "config.json")) as f:
        cfg = json.load(f)
    targets = cfg["quantization_config"]["config_groups"]["mtp_group"]["targets"]
    assert "mtp.embed_tokens" not in targets
    assert "mtp.layers.0.self_attn.q_proj" in targets


def test_quantize_and_save_mtp_keeps_fusion_proj_full_precision(tmp_path):
    """The MTP fusion projection (eh_proj / fc) is a plain nn.Linear with no
    scale param in some engines, so it must stay full precision and out of the
    config group even though it is a normal 2D linear weight."""
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    main = {"model.layers.0.weight": torch.randn(32, 32)}
    _make_source_checkpoint(src, main)
    _make_dest_checkpoint(dst, quant_config=FP8_QUANT_CONFIG)

    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp", mtp_scheme="FP8_DYNAMIC")

    from safetensors import safe_open

    with safe_open(os.path.join(dst, "model_mtp.safetensors"), framework="pt") as f:
        keys = set(f.keys())
    assert "mtp.eh_proj.weight" in keys
    assert "mtp.eh_proj.weight_scale" not in keys  # not quantized
    assert "mtp.layers.0.self_attn.q_proj.weight_scale" in keys  # linear is

    with open(os.path.join(dst, "config.json")) as f:
        cfg = json.load(f)
    targets = cfg["quantization_config"]["config_groups"]["mtp_group"]["targets"]
    assert "mtp.eh_proj" not in targets
    assert "mtp.layers.0.self_attn.q_proj" in targets


NVFP4_QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "float",
                "strategy": "tensor_group",
                "group_size": 16,
                "symmetric": True,
            },
        }
    },
    "format": "nvfp4-pack-quantized",
    "quantization_status": "compressed",
}


def _make_nvfp4_source(src, vocab_size=None):
    """Source checkpoint with a complete q/k/v fused set and a gate/up set."""
    mtp_tensors = {
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        "mtp.layers.0.self_attn.k_proj.weight": torch.randn(32, 32),
        "mtp.layers.0.self_attn.v_proj.weight": torch.randn(32, 32),
        "mtp.layers.0.self_attn.o_proj.weight": torch.randn(32, 32),
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(64, 32),
        "mtp.layers.0.mlp.up_proj.weight": torch.randn(64, 32),
        "mtp.eh_proj.weight": torch.randn(32, 64),
        "mtp.norm.weight": torch.randn(32),
    }
    if vocab_size is not None:
        mtp_tensors["mtp.embed_tokens.weight"] = torch.randn(vocab_size, 32)
    save_file(
        {"model.layers.0.weight": torch.randn(32, 32)},
        os.path.join(src, "model.safetensors"),
    )
    save_file(mtp_tensors, os.path.join(src, "model_mtp.safetensors"))
    weight_map = {"model.layers.0.weight": "model.safetensors"}
    weight_map.update({k: "model_mtp.safetensors" for k in mtp_tensors})
    with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return src


def test_quantize_and_save_mtp_nvfp4_fused_shared_global_scale(tmp_path):
    """NVFP4 (microscale) MTP linears are packed to fp4 with per-block scales
    and a per-tensor global scale; the q/k/v fused set shares one global scale
    so vLLM's fused QKV linear loads consistently. eh_proj stays full precision.
    """
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    _make_nvfp4_source(src)
    _make_dest_checkpoint(dst, quant_config=NVFP4_QUANT_CONFIG)

    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp", mtp_scheme="NVFP4")

    from safetensors import safe_open

    with safe_open(os.path.join(dst, "model_mtp.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        # linear weights are packed to fp4 (weight_packed) with scales
        assert "mtp.layers.0.self_attn.q_proj.weight_packed" in keys
        assert "mtp.layers.0.self_attn.q_proj.weight_scale" in keys
        assert "mtp.layers.0.self_attn.q_proj.weight_global_scale" in keys
        # NVFP4's local-dynamic input activations require a calibrated
        # input_global_scale, which MTP cannot produce -> dropped to weight-only
        assert not any(k.endswith("input_global_scale") for k in keys)
        # fusion proj + norm stay full precision, no packing
        assert "mtp.eh_proj.weight" in keys
        assert "mtp.eh_proj.weight_packed" not in keys
        assert "mtp.norm.weight" in keys
        # q/k/v share one global scale (fused-set coordination)
        gq = f.get_tensor("mtp.layers.0.self_attn.q_proj.weight_global_scale")
        gk = f.get_tensor("mtp.layers.0.self_attn.k_proj.weight_global_scale")
        gv = f.get_tensor("mtp.layers.0.self_attn.v_proj.weight_global_scale")
        assert torch.equal(gq, gk)
        assert torch.equal(gq, gv)
        # gate/up likewise share a global scale
        gg = f.get_tensor("mtp.layers.0.mlp.gate_proj.weight_global_scale")
        gu = f.get_tensor("mtp.layers.0.mlp.up_proj.weight_global_scale")
        assert torch.equal(gg, gu)

    with open(os.path.join(dst, "config.json")) as f:
        cfg = json.load(f)
    group = cfg["quantization_config"]["config_groups"]["mtp_group"]
    assert group["format"] == "nvfp4-pack-quantized"
    assert group["weights"]["num_bits"] == 4
    assert str(group["weights"]["strategy"]).endswith("tensor_group")
    targets = group["targets"]
    assert "mtp.layers.0.self_attn.q_proj" in targets
    assert "mtp.eh_proj" not in targets
    assert "mtp.norm" not in targets


def test_quantize_and_save_mtp_missing_shard_skips_gracefully(tmp_path):
    """A referenced MTP shard that is absent locally is skipped, not fatal."""
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    main = {"model.layers.0.weight": torch.randn(32, 32)}
    _make_source_checkpoint(src, main)
    _make_dest_checkpoint(dst, quant_config=FP8_QUANT_CONFIG)
    # remove the mtp shard so the index references a missing file
    os.remove(os.path.join(src, "model_mtp.safetensors"))

    # no local shard and no hub access -> no MTP tensors -> ValueError, not crash
    with pytest.raises(ValueError, match="No tensors with prefix"):
        _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp")


def test_quantize_and_save_mtp_unquantized_fallback_adds_ignore(tmp_path):
    """When no quantization_config is present, MTP tensors are saved unquantized
    and marked as ignored so inference engines skip them."""
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    main = {"model.layers.0.weight": torch.randn(32, 32)}
    _make_source_checkpoint(src, main)
    # dest has a quantization_config but with no config_groups → scheme is None
    _make_dest_checkpoint(dst, quant_config={"format": "dense"})

    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp")

    with open(os.path.join(dst, "config.json")) as f:
        cfg = json.load(f)
    ignore = cfg.get("quantization_config", {}).get("ignore", [])
    assert any("mtp" in s for s in ignore)


def test_quantize_and_save_mtp_defaults_to_bf16_and_ignores(tmp_path):
    """MTP quantization is opt-in: with no mtp_scheme the layers stay full
    precision (bf16) and are added to the ignore list, even when the main model
    is quantized. This is the default so save_pretrained is backwards compatible.
    """
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    main = {"model.layers.0.weight": torch.randn(32, 32)}
    _make_source_checkpoint(src, main)
    _make_dest_checkpoint(dst, quant_config=FP8_QUANT_CONFIG)

    # no mtp_scheme -> default (bf16), even though the main model is FP8
    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp")

    from safetensors import safe_open

    with safe_open(os.path.join(dst, "model_mtp.safetensors"), framework="pt") as f:
        keys = set(f.keys())
    # nothing is quantized: the linear keeps a plain .weight, gains no scale
    assert "mtp.layers.0.self_attn.q_proj.weight" in keys
    assert "mtp.layers.0.self_attn.q_proj.weight_scale" not in keys

    with open(os.path.join(dst, "config.json")) as f:
        cfg = json.load(f)
    quant_cfg = cfg.get("quantization_config", {})
    # no mtp_group is emitted, and MTP is marked ignored instead
    assert "mtp_group" not in quant_cfg.get("config_groups", {})
    assert any("mtp" in s for s in quant_cfg.get("ignore", []))


def test_quantize_and_save_mtp_explicit_preset_overrides_main_scheme(tmp_path):
    """An explicit preset name (e.g. "NVFP4") quantizes the MTP layers with that
    scheme regardless of the main model's scheme (here FP8)."""
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    _make_nvfp4_source(src)
    # main model is FP8, but we explicitly ask for NVFP4 on the MTP layers
    _make_dest_checkpoint(dst, quant_config=FP8_QUANT_CONFIG)

    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp", mtp_scheme="NVFP4")

    from safetensors import safe_open

    with safe_open(os.path.join(dst, "model_mtp.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        # NVFP4 packing, not FP8 -> weight_packed + global scale present
        assert "mtp.layers.0.self_attn.q_proj.weight_packed" in keys
        assert "mtp.layers.0.self_attn.q_proj.weight_global_scale" in keys

    with open(os.path.join(dst, "config.json")) as f:
        cfg = json.load(f)
    group = cfg["quantization_config"]["config_groups"]["mtp_group"]
    assert group["format"] == "nvfp4-pack-quantized"
    assert group["weights"]["num_bits"] == 4


def test_quantize_and_save_mtp_explicit_scheme_object(tmp_path):
    """A QuantizationScheme object is accepted directly as mtp_scheme."""
    from compressed_tensors.quantization import preset_name_to_scheme

    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    os.makedirs(src)
    os.makedirs(dst)
    main = {"model.layers.0.weight": torch.randn(32, 32)}
    _make_source_checkpoint(src, main)
    # no quantization_config at all on the dest model
    _make_dest_checkpoint(dst, quant_config=None)

    scheme = preset_name_to_scheme("FP8_DYNAMIC", targets=["re:.*\\.weight"])
    _quantize_and_save_mtp_tensors(src, dst, mtp_prefix="mtp", mtp_scheme=scheme)

    from safetensors import safe_open

    with safe_open(os.path.join(dst, "model_mtp.safetensors"), framework="pt") as f:
        keys = set(f.keys())
    assert "mtp.layers.0.self_attn.q_proj.weight_scale" in keys


# ---------------------------------------------------------------------------
# _resolve_mtp_scheme
# ---------------------------------------------------------------------------


def test_resolve_mtp_scheme_none_and_aliases():
    assert _resolve_mtp_scheme(None) is None
    for alias in ("bf16", "BF16", "bfloat16", "none", "dense", "unquantized"):
        assert _resolve_mtp_scheme(alias) is None


def test_resolve_mtp_scheme_preset_name():
    scheme = _resolve_mtp_scheme("NVFP4")
    assert scheme is not None
    assert scheme.weights.num_bits == 4
    assert str(scheme.weights.strategy).endswith("tensor_group")


def test_resolve_mtp_scheme_drops_local_dynamic_input_activations():
    """NVFP4 input activations use dynamic="local": per-group micro-scales are
    dynamic but a per-tensor input_global_scale is static and must be
    calibrated. MTP layers are never observed, so it is dropped to weight-only
    (leaving it in would emit a meaningless input_global_scale -> 1/scale=inf).
    """
    scheme = _resolve_mtp_scheme("NVFP4")
    assert scheme.weights is not None  # weights stay quantized (4-bit)
    assert scheme.input_activations is None  # local-dynamic acts dropped


def test_resolve_mtp_scheme_passthrough_weight_only_object():
    """A QuantizationScheme with no static activation quant is returned as-is."""
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    obj = QuantizationScheme(
        targets=["re:.*\\.weight"],
        weights=QuantizationArgs(
            num_bits=4, type="int", strategy="group", group_size=128
        ),
    )
    assert obj.input_activations is None
    assert _resolve_mtp_scheme(obj) is obj


def test_resolve_mtp_scheme_drops_static_input_activations():
    """Static (non-dynamic) input-activation quant cannot be calibrated for MTP
    layers, so it is dropped to weight-only regardless of how it was supplied."""
    # "FP8" preset uses static per-tensor input activations
    scheme = _resolve_mtp_scheme("FP8")
    assert scheme is not None
    assert scheme.weights is not None  # weights stay quantized
    assert scheme.input_activations is None  # static acts dropped


def test_resolve_mtp_scheme_keeps_dynamic_input_activations():
    """Dynamic input-activation quant needs no calibration, so it is kept."""
    scheme = _resolve_mtp_scheme("FP8_DYNAMIC")
    assert scheme is not None
    assert scheme.input_activations is not None
    assert scheme.input_activations.dynamic is True


def test_resolve_mtp_scheme_invalid_type_raises():
    with pytest.raises(TypeError, match="mtp_scheme must be"):
        _resolve_mtp_scheme(123)
