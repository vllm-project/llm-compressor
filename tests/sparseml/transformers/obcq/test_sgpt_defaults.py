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
class TestSGPTDefaults(unittest.TestCase):
    def test_sgpt_defaults(self):
        from sparseml.core.framework import Framework
        from sparseml.core.state import State
        from sparseml.modifiers.obcq import SparseGPTModifier

        kwargs = {"sparsity": 0.5}
        sparsegpt_modifier_only_sparsity = SparseGPTModifier(
            framework=Framework.pytorch, **kwargs
        )
        self.assertEqual(sparsegpt_modifier_only_sparsity.block_size, 128)
        self.assertEqual(sparsegpt_modifier_only_sparsity.sparsity, 0.5)

        # fail if we don't pass a sparsity or enable quantization
        kwargs = {}
        sparsegpt_invalid = SparseGPTModifier(framework=Framework.pytorch, **kwargs)
        state_test = State(framework=Framework.pytorch)
        sparsegpt_invalid.initialized_structure_ = True
        with self.assertRaises(ValueError):
            sparsegpt_invalid.on_initialize(state=state_test)
