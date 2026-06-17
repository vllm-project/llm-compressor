from unittest.mock import MagicMock

from compressed_tensors.quantization.quant_config import QuantizationConfig
from transformers.conversion_mapping import WeightRenaming

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
