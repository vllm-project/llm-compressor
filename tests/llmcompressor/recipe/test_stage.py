from llmcompressor.recipe import Recipe, StageRunType


def test_run_type_as_param():
    recipe_str = """
    first_stage:
        run_type: oneshot
        some_modifiers:
            QuantizationModifier:
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        targets: ["Linear"]
                        weights:
                            num_bits: 8
    second_stage:
        run_type: train
        some_modifiers:
            ConstantPruningModifier:
                start: 0.0
    """

    recipe = Recipe.create_instance(recipe_str)
    assert recipe.stages[0].infer_run_type() == StageRunType.ONESHOT
    assert recipe.stages[1].infer_run_type() == StageRunType.TRAIN


def test_run_type_as_name():
    recipe_str = """
    first_oneshot_stage:
        some_modifiers:
            QuantizationModifier:
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        targets: ["Linear"]
                        weights:
                            num_bits: 8
    second_train_stage:
        some_modifiers:
            ConstantPruningModifier:
                start: 0.0
    """

    recipe = Recipe.create_instance(recipe_str)
    assert recipe.stages[0].infer_run_type() == StageRunType.ONESHOT
    assert recipe.stages[1].infer_run_type() == StageRunType.TRAIN


def test_no_run_type():
    recipe_str = """
    first_stage:
        some_modifiers:
            QuantizationModifier:
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        targets: ["Linear"]
                        weights:
                            num_bits: 8
    second_stage:
        some_modifiers:
            ConstantPruningModifier:
                start: 0.0
    """

    recipe = Recipe.create_instance(recipe_str)
    assert recipe.stages[0].infer_run_type() is None
    assert recipe.stages[1].infer_run_type() is None
