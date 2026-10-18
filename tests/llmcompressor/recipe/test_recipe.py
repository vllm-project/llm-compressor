import tempfile
import warnings

import pytest
import yaml

# Suppress deprecation warning for AWQModifier import as this test
# explicitly tests backward compatibility of the deprecated API
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from llmcompressor.modifiers.awq import AWQModifier

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier as AWQTransformModifier
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

    expected_dict = recipe_instance.to_dict()
    actual_dict = recipe_from_serialized.to_dict()

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


@pytest.mark.parametrize(
    "modifiers,is_valid",
    [
        (
            [
                AWQModifier(
                    ignore=["lm_head"],
                    scheme="W4A16_ASYM",
                    targets=["Linear"],
                    duo_scaling="both",
                )
            ],
            True,
        ),
        (
            AWQModifier(
                ignore=["lm_head"],
                scheme="W4A16_ASYM",
                targets=["Linear"],
                duo_scaling="both",
            ),
            True,
        ),
        (
            [
                AWQTransformModifier(
                    duo_scaling="both",
                )
            ],
            False,
        ),
        (
            AWQTransformModifier(
                duo_scaling="both",
            ),
            False,
        ),
        (
            [
                AWQTransformModifier(
                    duo_scaling="both",
                ),
                QuantizationModifier(
                    ignore=["lm_head"],
                    scheme="W4A16_ASYM",
                    targets=["Linear"],
                ),
            ],
            True,
        ),
        (
            [
                QuantizationModifier(
                    ignore=["lm_head"],
                    scheme="W4A16_ASYM",
                    targets=["Linear"],
                ),
                AWQTransformModifier(
                    duo_scaling="both",
                ),
            ],
            False,
        ),
        (
            [
                AWQTransformModifier(
                    duo_scaling="both",
                ),
                GPTQModifier(
                    ignore=["lm_head"],
                    scheme="W4A16_ASYM",
                    targets=["Linear"],
                ),
            ],
            True,
        ),
    ],
)
def test_recipe_validate_after(modifiers, is_valid):
    if is_valid:
        Recipe.create_instance(modifiers)
    else:
        with pytest.raises(ValueError):
            Recipe.create_instance(modifiers)


def test_recipe_from_dict_loads_numbered_stages():
    from llmcompressor.recipe.utils import append_recipe_dict

    first = Recipe.from_modifiers(
        QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC")
    )
    second = Recipe.from_modifiers(
        QuantizationModifier(targets="Linear", scheme="W8A8")
    )

    # recipes merged by stacked oneshot runs use numbered stage keys
    merged = append_recipe_dict(first.to_dict(), second.to_dict())
    assert list(merged.keys()) == ["default_stage_0", "default_stage_1"]

    recipe = Recipe.from_dict(merged)
    assert recipe.stage == "default"
    assert [modifier.scheme for modifier in recipe.modifiers] == [
        "FP8_DYNAMIC",
        "W8A8",
    ]


def test_recipe_yaml_with_existing_recipe_round_trips(tmp_path):
    existing_path = tmp_path / "existing.yaml"
    Recipe.from_modifiers(
        QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC")
    ).yaml(file_path=str(existing_path))

    merged_path = tmp_path / "merged.yaml"
    Recipe.from_modifiers(QuantizationModifier(targets="Linear", scheme="W8A8")).yaml(
        file_path=str(merged_path), existing_recipe_path=str(existing_path)
    )

    recipe = Recipe.create_instance(str(merged_path))
    assert [modifier.scheme for modifier in recipe.modifiers] == [
        "FP8_DYNAMIC",
        "W8A8",
    ]


@pytest.mark.parametrize("stage_key", ["test_stage_name", "test_stage_"])
def test_recipe_from_dict_ignores_non_stage_keys(stage_key):
    recipe_dict = {
        stage_key: {
            "quant_modifiers": {
                "QuantizationModifier": {"targets": "Linear", "scheme": "W8A8"}
            }
        }
    }

    assert Recipe.from_dict(recipe_dict).modifiers == []
