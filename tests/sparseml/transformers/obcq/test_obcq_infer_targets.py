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

import unittest

import pytest

from tests.testing_utils import requires_torch


@pytest.mark.integration
@requires_torch
class TestInferTargets(unittest.TestCase):
    def setUp(self):
        from sparseml.core import ModifiableModel
        from sparseml.core.framework import Framework
        from sparseml.transformers import SparseAutoModelForCausalLM

        model = SparseAutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M")
        self.modifiable_model = ModifiableModel(
            framework=Framework.pytorch, model=model
        )
        self.targets = self.modifiable_model.get_no_split_params()

    def test_infer_targets(self):
        from sparseml.modifiers.obcq.pytorch import SparseGPTModifierPyTorch

        self.assertEqual(len(self.targets), 1)
        self.assertEqual(self.targets[0], "LlamaDecoderLayer")

        modifier = SparseGPTModifierPyTorch(sparsity=0.5)
        modifier.targets = self.targets
        modifier.model = self.modifiable_model
        compressible_layers = modifier.compressible_layers()
        self.assertEqual(len(compressible_layers), 6)
