import json
from unittest.mock import MagicMock

import torch
from compressed_tensors.quantization.quant_config import QuantizationConfig
from transformers import AutoModelForImageTextToText
from transformers.conversion_mapping import WeightRenaming

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    _map_ignore_to_checkpoint_names,
)


def _model_with_conversions(conversions):
    """Create a mock model with _weight_conversions set."""
    model = MagicMock()
    model._weight_conversions = conversions
    return model


def _make_quant_config(ignore):
    """Create a minimal QuantizationConfig with the given ignore list."""
    return QuantizationConfig(
        config_groups={},
        ignore=ignore,
    )


class TestMapIgnoreToCheckpointNames:
    def test_multiple_independent_renamings(self):
        conversions = [
            WeightRenaming("vision_embedder\\.patch_dense", "embed_vision.patch_dense"),
            WeightRenaming("vision_embedder\\.pos_norm", "embed_vision.pos_norm"),
            WeightRenaming(
                "embed_vision\\.embedding_projection",
                "embed_vision.multimodal_embedder.embedding_projection",
            ),
        ]
        model = _model_with_conversions(conversions)
        qconfig = _make_quant_config(
            [
                "model.embed_vision.patch_dense",
                "model.embed_vision.multimodal_embedder.embedding_projection",
                "model.embed_audio.embedding_projection",
                "lm_head",
            ]
        )

        _map_ignore_to_checkpoint_names(model, qconfig)

        assert qconfig.ignore == [
            "model.vision_embedder.patch_dense",
            "model.embed_vision.embedding_projection",
            "model.embed_audio.embedding_projection",
            "lm_head",
        ]

    def test_chained_renamings(self):
        conversions = [
            WeightRenaming("old_prefix", "mid_prefix"),
            WeightRenaming("mid_prefix\\.old_leaf", "mid_prefix.new_leaf"),
        ]
        model = _model_with_conversions(conversions)
        qconfig = _make_quant_config(["model.mid_prefix.new_leaf"])

        _map_ignore_to_checkpoint_names(model, qconfig)

        assert qconfig.ignore == ["model.old_prefix.old_leaf"]

    def test_no_conversions(self):
        model = _model_with_conversions([])
        qconfig = _make_quant_config(["model.layer.0", "lm_head"])

        _map_ignore_to_checkpoint_names(model, qconfig)

        assert qconfig.ignore == ["model.layer.0", "lm_head"]

    def test_no_match_unchanged(self):
        conversions = [
            WeightRenaming("vision_embedder", "embed_vision"),
        ]
        model = _model_with_conversions(conversions)
        qconfig = _make_quant_config(["lm_head", "model.layers.0.mlp"])

        _map_ignore_to_checkpoint_names(model, qconfig)

        assert qconfig.ignore == ["lm_head", "model.layers.0.mlp"]

    def test_none_config(self):
        model = _model_with_conversions([])
        _map_ignore_to_checkpoint_names(model, None)

    def test_empty_ignore(self):
        conversions = [
            WeightRenaming("vision_embedder", "embed_vision"),
        ]
        model = _model_with_conversions(conversions)
        qconfig = _make_quant_config([])

        _map_ignore_to_checkpoint_names(model, qconfig)

        assert qconfig.ignore == []


def test_oneshot_ignore_mapped_to_checkpoint_names(tmp_path):
    """Quantize a real vision-language model with an ignore list and verify
    that the saved config contains checkpoint-style names, not HF module names.

    llava has WeightRenaming conversions — HF module names differ from
    checkpoint key names (e.g. ``model.vision_tower.encoder.layers.0.…``
    in HF vs ``vision_tower.vision_model.encoder.layers.0.…`` on disk).
    Without the reverse mapping, the saved ignore list would reference HF
    names that don't match any safetensors keys.
    """
    model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float16)

    ignore = [
        "lm_head",
        "model.vision_tower.encoder.layers.0.self_attn.q_proj",
        "model.multi_modal_projector.linear_1",
    ]

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=ignore,
    )

    oneshot(model=model, recipe=recipe)

    save_dir = tmp_path / "saved"
    model.save_pretrained(save_dir)

    with open(save_dir / "config.json") as f:
        saved_config = json.load(f)

    saved_ignore = saved_config["quantization_config"]["ignore"]

    assert "lm_head" in saved_ignore
    assert "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj" in saved_ignore
    assert "multi_modal_projector.linear_1" in saved_ignore

    for entry in saved_ignore:
        assert not entry.startswith(
            "model."
        ), f"HF name leaked into saved config: {entry}"
