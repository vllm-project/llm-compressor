from src.llmcompressor.recipe import Recipe


def test_recipe_model_dump():
    """Test that model_dump produces a format compatible with model_validate."""
    # Create a recipe with multiple stages and modifiers
    recipe_str = """
    version: "1.0"
    args:
        learning_rate: 0.001
    train_stage:
        pruning_modifiers:
            ConstantPruningModifier:
                start: 0.0
                end: 2.0
                targets: ['re:.*weight']
        quantization_modifiers:
            QuantizationModifier:
                bits: 8
                targets: ['re:.*weight']
    eval_stage:
        pruning_modifiers:
            ConstantPruningModifier:
                start: 2.0
                end: 4.0
                targets: ['re:.*weight']
    """

    # Create recipe instance
    recipe = Recipe.create_instance(recipe_str)

    # Get dictionary representation
    recipe_dict = recipe.model_dump()

    # Verify the structure is compatible with model_validate
    # by creating a new recipe from the dictionary
    new_recipe = Recipe.model_validate(recipe_dict)

    # Verify version and args are preserved
    assert new_recipe.version == recipe.version
    assert new_recipe.args == recipe.args

    # Verify stages are preserved
    assert len(new_recipe.stages) == len(recipe.stages)

    # Verify stage names and modifiers are preserved
    for new_stage, orig_stage in zip(new_recipe.stages, recipe.stages):
        assert new_stage.group == orig_stage.group
        assert len(new_stage.modifiers) == len(orig_stage.modifiers)

        # Verify modifier types and args are preserved
        for new_mod, orig_mod in zip(new_stage.modifiers, orig_stage.modifiers):
            assert new_mod.type == orig_mod.type
            assert new_mod.args == orig_mod.args
