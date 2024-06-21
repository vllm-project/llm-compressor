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


def valid_recipe_strings():
    return [
        """
        test_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
                    targets: __ALL_PRUNABLE__
        """,
        """
        test_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
                    targets: __ALL_PRUNABLE__
                MagnitudePruningModifier:
                    start: 5
                    end: 10
                    init_sparsity: 0.1
                    final_sparsity: 0.5
                    targets: __ALL_PRUNABLE__
        """,
        """
        test1_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
                    targets: __ALL_PRUNABLE__
        test2_stage:
                MagnitudePruningModifier:
                    start: 5
                    end: 10
                    init_sparsity: 0.1
                    final_sparsity: 0.5
                    targets: __ALL_PRUNABLE__
        """,
        """
        test1_stage:
            constant_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
                    targets: __ALL_PRUNABLE__
            magnitude_modifiers:
                MagnitudePruningModifier:
                    start: 5
                    end: 10
                    init_sparsity: 0.1
                    final_sparsity: 0.5
                    targets: __ALL_PRUNABLE__
        """,
    ]
