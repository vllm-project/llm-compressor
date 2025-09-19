import tempfile

import pytest
import yaml

from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.recipe import Recipe
from tests.llmcompressor.helpers import valid_recipe_strings


@pytest.mark.parametrize("recipe_str", valid_recipe_strings())
def test_recipe_create_instance_accepts_valid_recipe_string(recipe_str):
    recipe = Recipe.create_instance(recipe_str)
    assert recipe is not None, "Recipe could not be created from string"


@pytest.mark.parametrize("recipe_str", valid_recipe_strings())
def test_recipe_create_instance_accepts_valid_recipe_file(recipe_str):
    content = yaml.safe_load(recipe_str)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(content, f)
        recipe = Recipe.create_instance(f.name)
        assert recipe is not None, "Recipe could not be created from file"


@pytest.mark.parametrize("recipe_str", valid_recipe_strings())
def test_serialization(recipe_str):
    recipe_instance = Recipe.create_instance(recipe_str)
    serialized_recipe = recipe_instance.yaml()
    recipe_from_serialized = Recipe.create_instance(serialized_recipe)

    expected_dict = recipe_instance.dict()
    actual_dict = recipe_from_serialized.dict()

    assert expected_dict == actual_dict


def test_recipe_creates_correct_modifier():
    start = 1
    end = 10
    targets = "__ALL_PRUNABLE__"

    yaml_str = f"""
        test_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: {start}
                    end: {end}
                    targets: {targets}
        """

    recipe_instance = Recipe.create_instance(yaml_str)

    stage_modifiers = recipe_instance.modifiers
    assert len(modifiers := stage_modifiers) == 1
    from llmcompressor.modifiers.pruning.constant import ConstantPruningModifier

    assert isinstance(modifier := modifiers[0], ConstantPruningModifier)
    assert modifier.start == start
    assert modifier.end == end


def test_recipe_can_be_created_from_modifier_instances():
    modifier = SparseGPTModifier(
        sparsity=0.5,
        group="pruning",
    )
    group_name = "dummy"

    # for pep8 compliance
    recipe_str = (
        f"{group_name}_stage:\n"
        "   pruning_modifiers:\n"
        "       SparseGPTModifier:\n"
        "           sparsity: 0.5\n"
    )

    expected_recipe_instance = Recipe.create_instance(recipe_str)
    expected_modifiers = expected_recipe_instance.modifiers

    actual_recipe_instance = Recipe.create_instance(
        [modifier], modifier_group_name=group_name
    )
    actual_modifiers = actual_recipe_instance.modifiers

    # assert num stages is the same
    assert len(actual_modifiers) == len(expected_modifiers)

    # assert num modifiers in each stage is the same
    assert len(actual_modifiers) == len(expected_modifiers)

    # assert modifiers in each stage are the same type
    # and have the same parameters
    for actual_modifier, expected_modifier in zip(actual_modifiers, expected_modifiers):
        assert isinstance(actual_modifier, type(expected_modifier))
        assert actual_modifier.model_dump() == expected_modifier.model_dump()
