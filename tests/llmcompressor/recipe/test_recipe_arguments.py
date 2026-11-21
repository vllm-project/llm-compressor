import pytest

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.recipe import Recipe


def test_recipe_with_an_explicit_group():
    # `group` is read out of the modifier arguments and passed explicitly, so
    # it must not be passed a second time through the arguments
    recipe = Recipe.create_instance(
        """
        test_stage:
            quantization_modifiers:
                QuantizationModifier:
                    scheme: FP8_DYNAMIC
                    group: my_group
        """
    )

    assert len(recipe.modifiers) == 1
    assert recipe.modifiers[0].group == "my_group"
    assert recipe.modifiers[0].scheme == "FP8_DYNAMIC"


def test_recipe_with_a_modifier_without_arguments():
    # a modifier without any arguments is written with an empty value, which
    # parses to None
    recipe = Recipe.create_instance(
        """
        test_stage:
            quantization_modifiers:
                QuantizationModifier:
        """
    )

    assert len(recipe.modifiers) == 1
    assert isinstance(recipe.modifiers[0], QuantizationModifier)
    assert recipe.modifiers[0].group == "quantization"


def test_recipe_with_invalid_modifier_arguments():
    # the arguments of a modifier have to be a mapping
    with pytest.raises(ValueError, match="must be a dictionary"):
        Recipe.create_instance(
            """
            test_stage:
                quantization_modifiers:
                    QuantizationModifier: "not a mapping"
            """
        )
