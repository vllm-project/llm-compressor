# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sparseml.core import Recipe
from sparseml.core.recipe import StageRunType


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
