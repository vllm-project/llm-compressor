from unittest.mock import MagicMock

from transformers.conversion_mapping import WeightRenaming

from llmcompressor.transformers.compression.compressed_tensors_utils import (
    _map_to_checkpoint_names,
)


def _model_with_conversions(conversions):
    """Create a mock model with _weight_conversions set."""
    model = MagicMock()
    model._weight_conversions = conversions
    return model


def _model_without_conversions():
    """Create a mock model with no _weight_conversions (no renaming needed)."""
    model = MagicMock(spec=[])
    return model


class TestMapToCheckpointNames:
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
        ignore = [
            "model.embed_vision.patch_dense",
            "model.embed_vision.multimodal_embedder.embedding_projection",
            "model.embed_audio.embedding_projection",
            "lm_head",
        ]

        result = _map_to_checkpoint_names(model, ignore)

        assert result == [
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
        ignore = ["model.mid_prefix.new_leaf"]

        result = _map_to_checkpoint_names(model, ignore)

        # Both renamings should fire in reverse order:
        # mid_prefix.new_leaf -> mid_prefix.old_leaf (reverse of conv 2)
        # mid_prefix.old_leaf -> old_prefix.old_leaf (reverse of conv 1)
        assert result == ["model.old_prefix.old_leaf"]

    def test_regex_pattern_single_match(self):
        """Test that regex patterns in ignore list get properly mapped."""
        conversions = [
            WeightRenaming("vision_embedder", "embed_vision"),
        ]
        model = _model_with_conversions(conversions)
        ignore = [r".*embed_vision\..*"]

        result = _map_to_checkpoint_names(model, ignore)

        # The regex pattern should be reversed to use checkpoint names
        assert result == [r".*vision_embedder\..*"]

    def test_regex_pattern_multiple_conversions(self):
        """Test regex patterns with multiple sequential conversions."""
        conversions = [
            WeightRenaming("vision_embedder", "embed_vision"),
            WeightRenaming("embed_vision\\.proj", "embed_vision.projection"),
        ]
        model = _model_with_conversions(conversions)
        ignore = [r"model\.embed_vision\.projection\..*"]

        result = _map_to_checkpoint_names(model, ignore)

        # Both conversions should be reversed
        assert result == [r"model\.vision_embedder\.proj\..*"]

    def test_regex_alternation_pattern(self):
        """Test regex patterns with alternation (|)."""
        conversions = [
            WeightRenaming("audio_embedder", "embed_audio"),
            WeightRenaming("vision_embedder", "embed_vision"),
        ]
        model = _model_with_conversions(conversions)
        ignore = [r".*(embed_audio|embed_vision)\..*"]

        result = _map_to_checkpoint_names(model, ignore)

        # The pattern should have both alternatives reversed
        assert result == [r".*(audio_embedder|vision_embedder)\..*"]

    def test_mixed_literal_and_regex(self):
        """Test a mix of literal names and regex patterns."""
        conversions = [
            WeightRenaming("vision_embedder", "embed_vision"),
        ]
        model = _model_with_conversions(conversions)
        ignore = [
            "model.embed_vision.layer1",  # literal
            r".*embed_vision\.layer[23].*",  # regex
        ]

        result = _map_to_checkpoint_names(model, ignore)

        assert result == [
            "model.vision_embedder.layer1",
            r".*vision_embedder\.layer[23].*",
        ]

    def test_no_match_regex_unchanged(self):
        """Test that regex patterns that don't match any conversion are unchanged."""
        conversions = [
            WeightRenaming("vision_embedder", "embed_vision"),
        ]
        model = _model_with_conversions(conversions)
        ignore = [r".*lm_head\..*"]

        result = _map_to_checkpoint_names(model, ignore)

        # Pattern doesn't match any conversion, should remain unchanged
        assert result == [r".*lm_head\..*"]
