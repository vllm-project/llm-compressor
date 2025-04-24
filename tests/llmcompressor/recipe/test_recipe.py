import os
import tempfile

import pytest
import yaml

from llmcompressor.modifiers.obcq.base import SparseGPTModifier
from llmcompressor.modifiers.pruning.constant import ConstantPruningModifier
from llmcompressor.recipe import Recipe
from tests.llmcompressor.helpers import valid_recipe_strings


@pytest.mark.parametrize("recipe_str", valid_recipe_strings())
class TestRecipeWithStrings:
    """Tests that use various recipe strings for validation."""

    def test_create_from_string(self, recipe_str):
        """Test creating a Recipe from a YAML string."""
        recipe = Recipe.create_instance(recipe_str)
        assert recipe is not None, "Recipe could not be created from string"
        assert isinstance(recipe, Recipe), "Created object is not a Recipe instance"

    def test_create_from_file(self, recipe_str):
        """Test creating a Recipe from a YAML file."""
        content = yaml.safe_load(recipe_str)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(content, f)
            f.flush()  # Ensure content is written
            recipe = Recipe.create_instance(f.name)
            assert recipe is not None, "Recipe could not be created from file"
            assert isinstance(recipe, Recipe), "Created object is not a Recipe instance"

    def test_yaml_serialization_roundtrip(self, recipe_str):
        """
        Test that a recipe can be serialized to YAML
        and deserialized back with all properties preserved.
        """
        # Create original recipe
        original_recipe = Recipe.create_instance(recipe_str)

        # Serialize to YAML
        yaml_str = original_recipe.yaml()
        assert yaml_str, "Serialized YAML string should not be empty"

        # Deserialize from YAML
        deserialized_recipe = Recipe.create_instance(yaml_str)

        # Compare serialized forms
        original_dict = original_recipe.model_dump()
        deserialized_dict = deserialized_recipe.model_dump()

        assert original_dict == deserialized_dict, "Serialization roundtrip failed"

    def test_model_dump_and_validate(self, recipe_str):
        """
        Test that model_dump produces a format compatible
        with model_validate.
        """
        recipe = Recipe.create_instance(recipe_str)
        validated_recipe = Recipe.model_validate(recipe.model_dump())
        assert (
            recipe == validated_recipe
        ), "Recipe instance and validated recipe do not match"


class TestRecipeSerialization:
    """
    Tests for Recipe serialization and deserialization
      edge cases."""

    def test_empty_recipe_serialization(self):
        """Test serialization of a minimal recipe with no stages."""
        recipe = Recipe()
        assert len(recipe.stages) == 0, "New recipe should have no stages"

        # Test roundtrip serialization
        dumped = recipe.model_dump()
        loaded = Recipe.model_validate(dumped)
        assert recipe == loaded, "Empty recipe serialization failed"

    def test_file_serialization(self):
        """Test serializing a recipe to a file and reading it back."""
        recipe = Recipe.create_instance(valid_recipe_strings()[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "recipe.yaml")

            # Write to file
            recipe.yaml(file_path=file_path)
            assert os.path.exists(file_path), "YAML file was not created"
            assert os.path.getsize(file_path) > 0, "YAML file is empty"

            # Read back from file
            loaded_recipe = Recipe.create_instance(file_path)
            assert (
                recipe == loaded_recipe
            ), "Recipe loaded from file doesn't match original"


class TestRecipeModifiers:
    """Tests for creating and working with modifiers in recipes."""

    def test_creates_correct_modifier(self):
        """
        Test that a recipe creates the expected modifier type
        with correct parameters.
        """
        # Recipe parameters
        params = {"start": 1, "end": 10, "targets": "__ALL_PRUNABLE__"}

        # Create recipe from YAML
        yaml_str = f"""
            test_stage:
                pruning_modifiers:
                    ConstantPruningModifier:
                        start: {params['start']}
                        end: {params['end']}
                        targets: {params['targets']}
            """
        recipe = Recipe.create_instance(yaml_str)

        # Get modifiers from recipe
        stage_modifiers = recipe.create_modifier()
        assert len(stage_modifiers) == 1, "Expected exactly one stage modifier"

        modifiers = stage_modifiers[0].modifiers
        assert len(modifiers) == 1, "Expected exactly one modifier in the stage"

        # Verify modifier type and parameters
        modifier = modifiers[0]
        assert isinstance(
            modifier, ConstantPruningModifier
        ), "Wrong modifier type created"
        assert modifier.start == params["start"], "Modifier start value incorrect"
        assert modifier.end == params["end"], "Modifier end value incorrect"
        assert modifier.targets == params["targets"], "Modifier targets incorrect"

    def test_create_from_modifier_instances(self):
        """Test creating a recipe from modifier instances."""
        # Create a modifier instance
        sparsity_value = 0.5
        modifier = SparseGPTModifier(sparsity=sparsity_value)
        group_name = "dummy"

        # Expected YAML representation
        recipe_str = (
            f"{group_name}_stage:\n"
            "   pruning_modifiers:\n"
            "       SparseGPTModifier:\n"
            f"           sparsity: {sparsity_value}\n"
        )

        # Create recipes for comparison
        expected_recipe = Recipe.create_instance(recipe_str)
        actual_recipe = Recipe.create_instance(
            [modifier], modifier_group_name=group_name
        )

        # Compare recipes by creating and checking their modifiers
        self._compare_recipe_modifiers(actual_recipe, expected_recipe)

    def _compare_recipe_modifiers(self, actual_recipe, expected_recipe):
        """Helper method to compare modifiers created from two recipes."""
        actual_modifiers = actual_recipe.create_modifier()
        expected_modifiers = expected_recipe.create_modifier()

        # Compare stage counts
        assert len(actual_modifiers) == len(expected_modifiers), "Stage counts differ"

        if not actual_modifiers:
            return  # No modifiers to compare

        # Compare modifier counts in each stage
        assert len(actual_modifiers[0].modifiers) == len(
            expected_modifiers[0].modifiers
        ), "Modifier counts differ"

        # Compare modifier types and parameters
        for actual_mod, expected_mod in zip(
            actual_modifiers[0].modifiers, expected_modifiers[0].modifiers
        ):
            assert isinstance(actual_mod, type(expected_mod)), "Modifier types differ"
            assert (
                actual_mod.model_dump() == expected_mod.model_dump()
            ), "Modifier parameters differ"
