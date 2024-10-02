import tempfile

import pytest
import yaml

from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.obcq.base import SparseGPTModifier
from llmcompressor.recipe import Recipe
from llmcompressor.recipe.recipe import create_recipe_string_from_modifiers
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

    stage_modifiers = recipe_instance.create_modifier()
    assert len(stage_modifiers) == 1
    assert len(modifiers := stage_modifiers[0].modifiers) == 1
    from llmcompressor.modifiers.pruning.constant import ConstantPruningModifier

    assert isinstance(modifier := modifiers[0], ConstantPruningModifier)
    assert modifier.start == start
    assert modifier.end == end


def test_recipe_can_be_created_from_modifier_instances():
    modifier = SparseGPTModifier(
        sparsity=0.5,
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
    expected_modifiers = expected_recipe_instance.create_modifier()

    actual_recipe_instance = Recipe.create_instance(
        [modifier], modifier_group_name=group_name
    )
    actual_modifiers = actual_recipe_instance.create_modifier()

    # assert num stages is the same
    assert len(actual_modifiers) == len(expected_modifiers)

    # assert num modifiers in each stage is the same
    assert len(actual_modifiers[0].modifiers) == len(expected_modifiers[0].modifiers)

    # assert modifiers in each stage are the same type
    # and have the same parameters
    for actual_modifier, expected_modifier in zip(
        actual_modifiers[0].modifiers, expected_modifiers[0].modifiers
    ):
        assert isinstance(actual_modifier, type(expected_modifier))
        assert actual_modifier.dict() == expected_modifier.dict()


class A_FirstDummyModifier(Modifier):
    def on_initialize(self, *args, **kwargs) -> bool:
        return True


class B_SecondDummyModifier(Modifier):
    def on_initialize(self, *args, **kwargs) -> bool:
        return True


def test_create_recipe_string_from_modifiers_with_default_group_name():
    modifiers = [B_SecondDummyModifier(), A_FirstDummyModifier()]
    expected_recipe_str = (
        "DEFAULT_stage:\n"
        "  DEFAULT_modifiers:\n"
        "    B_SecondDummyModifier: {}\n"
        "    A_FirstDummyModifier: {}\n"
    )
    actual_recipe_str = create_recipe_string_from_modifiers(modifiers)
    assert actual_recipe_str == expected_recipe_str


def test_create_recipe_string_from_modifiers_with_custom_group_name():
    modifiers = [B_SecondDummyModifier(), A_FirstDummyModifier()]
    group_name = "custom"
    expected_recipe_str = (
        "custom_stage:\n"
        "  DEFAULT_modifiers:\n"
        "    B_SecondDummyModifier: {}\n"
        "    A_FirstDummyModifier: {}\n"
    )
    actual_recipe_str = create_recipe_string_from_modifiers(modifiers, group_name)
    assert actual_recipe_str == expected_recipe_str


def test_create_recipe_string_from_modifiers_with_empty_modifiers():
    modifiers = []
    expected_recipe_str = "DEFAULT_stage:\n" "  DEFAULT_modifiers: {}\n"
    actual_recipe_str = create_recipe_string_from_modifiers(modifiers)
    assert actual_recipe_str == expected_recipe_str
