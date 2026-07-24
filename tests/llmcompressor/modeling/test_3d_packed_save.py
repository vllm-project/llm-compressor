"""Tests for 3D packed MoE save mappings (issue #2699)."""

from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration
from transformers.core_model_loading import revert_weight_conversion

from llmcompressor.modeling.moe.conversion_mappings import (
    ARCH_TO_2D_MAPPINGS,
    get_linearize_load_mappings,
    has_linearize_load_mappings,
    set_linearize_save_mappings,
)
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.utils.dev import skip_weights_initialize


def _tiny_qwen3_vl_moe():
    config = Qwen3VLMoeConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "vocab_size": 256,
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
        },
    )
    with skip_weights_initialize():
        model = Qwen3VLMoeForConditionalGeneration(config)
    return model


def test_has_linearize_load_mappings_for_qwen3_vl_moe():
    assert has_linearize_load_mappings("qwen3_vl_moe")
    assert "qwen3_vl_moe" in ARCH_TO_2D_MAPPINGS
    assert "qwen3_vl_moe_text" in ARCH_TO_2D_MAPPINGS
    assert not has_linearize_load_mappings("mixtral")


def test_set_linearize_save_mappings_dict_config():
    class _Model:
        config = {
            "model_type": "qwen3_vl_moe",
            "text_config": {"model_type": "qwen3_vl_moe_text"},
        }

    model = _Model()
    assert set_linearize_save_mappings(model)
    assert getattr(model, "_weight_conversions", None)
    assert len(model._weight_conversions) > 0


def test_explicit_backwards_mapping_is_unpack_not_reversed():
    """Tuple 3rd entry is the unpack converters used as-is (HF reverses on save)."""
    remove_targets, load_extra, backwards = ARCH_TO_2D_MAPPINGS["qwen3_vl_moe"]
    assert remove_targets == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    assert backwards is not None
    assert backwards is load_extra
    assert any(
        "gate_up_proj" in getattr(converter, "source_patterns", [""])[0]
        for converter in backwards
    )

    _experts_cls, load_map, save_map = get_linearize_load_mappings("qwen3_vl_moe")
    assert save_map is backwards
    assert all(converter in load_map for converter in save_map)


def test_qwen2_moe_stays_2d_on_save():
    _remove_targets, _load_extra, backwards = ARCH_TO_2D_MAPPINGS["qwen2_moe"]
    assert backwards is None


def test_revert_packs_linearized_weights_and_scales():
    num_experts, moe_intermediate, hidden = 4, 8, 16
    _experts_cls, _load_map, save_map = get_linearize_load_mappings("qwen3_vl_moe")

    state_dict = {}
    for expert_idx in range(num_experts):
        state_dict[
            f"model.language_model.layers.0.mlp.experts.{expert_idx}.gate_proj.weight"
        ] = torch.randn(moe_intermediate, hidden)
        state_dict[
            f"model.language_model.layers.0.mlp.experts.{expert_idx}.up_proj.weight"
        ] = torch.randn(moe_intermediate, hidden)
        state_dict[
            f"model.language_model.layers.0.mlp.experts.{expert_idx}.down_proj.weight"
        ] = torch.randn(hidden, moe_intermediate)
        state_dict[
            f"model.language_model.layers.0.mlp.experts.{expert_idx}.gate_proj.weight_scale"
        ] = torch.randn(moe_intermediate)
        state_dict[
            f"model.language_model.layers.0.mlp.experts.{expert_idx}.up_proj.weight_scale"
        ] = torch.randn(moe_intermediate)
        state_dict[
            f"model.language_model.layers.0.mlp.experts.{expert_idx}.down_proj.weight_scale"
        ] = torch.randn(hidden)

    class _Model:
        config = None
        _weight_conversions = save_map

        def get_parameter(self, name):
            raise RuntimeError(name)

    packed = revert_weight_conversion(_Model(), state_dict)
    expert_keys = sorted(k for k in packed if "experts" in k)

    assert expert_keys == [
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.mlp.experts.down_proj_scale",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.gate_up_proj_scale",
    ]
    # Disk layout after inverse of unpack: gate_up [E, H, 2I], down [E, I, H]
    assert packed["model.language_model.layers.0.mlp.experts.gate_up_proj"].shape == (
        num_experts,
        hidden,
        2 * moe_intermediate,
    )
    assert packed["model.language_model.layers.0.mlp.experts.down_proj"].shape == (
        num_experts,
        moe_intermediate,
        hidden,
    )
    assert packed[
        "model.language_model.layers.0.mlp.experts.gate_up_proj_scale"
    ].shape == (num_experts, 2 * moe_intermediate)
    assert packed[
        "model.language_model.layers.0.mlp.experts.down_proj_scale"
    ].shape == (
        num_experts,
        hidden,
    )


def test_linearize_installs_3d_save_mappings():
    model = _tiny_qwen3_vl_moe()
    experts = model.model.language_model.layers[0].mlp.experts
    assert not isinstance(experts, LinearExperts2D)

    linearize_moe(model)
    experts = model.model.language_model.layers[0].mlp.experts
    assert isinstance(experts, LinearExperts2D)
    assert getattr(model, "_weight_conversions", None)
    assert len(model._weight_conversions) > 0


def test_save_pretrained_writes_3d_packed_experts(tmp_path: Path):
    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)

    out_dir = tmp_path / "qwen3_vl_moe_packed"
    model.save_pretrained(out_dir, safe_serialization=True)

    weight_files = list(out_dir.glob("*.safetensors"))
    assert weight_files, "expected safetensors output"

    expert_keys = []
    for path in weight_files:
        with safe_open(path, framework="pt") as handle:
            expert_keys.extend(k for k in handle.keys() if "mlp.experts" in k)

    assert any(k.endswith("experts.gate_up_proj") for k in expert_keys)
    assert any(k.endswith("experts.down_proj") for k in expert_keys)
    assert not any(".experts.0." in k for k in expert_keys)


def test_transformers_reload_3d_packed(tmp_path: Path):
    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    out_dir = tmp_path / "reload"
    model.save_pretrained(out_dir, safe_serialization=True)

    reloaded = Qwen3VLMoeForConditionalGeneration.from_pretrained(out_dir)
    experts = reloaded.model.language_model.layers[0].mlp.experts
    assert hasattr(experts, "gate_up_proj")
    assert hasattr(experts, "down_proj")
    assert not isinstance(experts, LinearExperts2D)
