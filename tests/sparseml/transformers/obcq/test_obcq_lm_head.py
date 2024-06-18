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
class TestLMHead(unittest.TestCase):
    def setUp(self):
        import torch

        from sparseml.transformers import SparseAutoModelForCausalLM

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = SparseAutoModelForCausalLM.from_pretrained(
            "Xenova/llama2.c-stories15M", device_map=self.device
        )
        self.kwargs = {
            "sparsity": 0.5,
            "block_size": 128,
            "quantize": False,
            "targets": [
                "model.layers.0",
                "model.layers.1",
                "model.layers.2",
                "model.layers.3",
                "model.layers.4",
                "model.layers.5",
            ],
        }

    def test_lm_head_target(self):
        from sparseml.core.framework import Framework
        from sparseml.core.state import State
        from sparseml.modifiers.obcq import SparseGPTModifier

        sparsegpt_modifier_no_head = SparseGPTModifier(
            framework=Framework.pytorch, **self.kwargs
        )

        state = State(framework=Framework.pytorch)
        state.update(model=self.model, device=self.device)
        sparsegpt_modifier_no_head.initialize_compression(state.model)

        self.kwargs["targets"].append("lm_head")
        sparsegpt_modifier_head = SparseGPTModifier(
            framework=Framework.pytorch, **self.kwargs
        )
        sparsegpt_modifier_head.initialize_compression(state.model)

        # check we pick up the lm_head layer
        layers_no_head = len(sparsegpt_modifier_no_head.compressible_layers_)
        layers_head = len(sparsegpt_modifier_head.compressible_layers_)
        self.assertEqual(layers_head, layers_no_head + 1)
