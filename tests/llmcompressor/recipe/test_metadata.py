import pytest

from llmcompressor.recipe.metadata import ModelMetaData, RecipeMetaData


class TestRecipeMetaData:
    @pytest.mark.parametrize(
        "self_metadata",
        [
            dict(domain="cv", task="classification"),
            dict(),
        ],
    )
    @pytest.mark.parametrize(
        "other_metadata",
        [
            dict(domain="domain", task="segmentation", requirements=["torch>=1.6.0"]),
            dict(
                domain="cv",
                task="task",
                target_model=ModelMetaData(layer_prefix="something"),
            ),
        ],
    )
    def test_update_missing_metadata(self, self_metadata, other_metadata):
        metadata_a = RecipeMetaData(**self_metadata)
        metadata_b = RecipeMetaData(**other_metadata)

        metadata_a.update_missing_metadata(metadata_b)

        all_keys = set(self_metadata.keys()).union(other_metadata.keys())

        # keys should not be overwritten
        # if they already exist
        for key in all_keys:
            if key in self_metadata:
                assert getattr(metadata_a, key) == self_metadata[key]
            elif key in other_metadata:
                assert getattr(metadata_a, key) == other_metadata[key]
