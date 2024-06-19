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
from torch.nn import Linear

from sparseml.core import State
from sparseml.core.framework import Framework
from sparseml.core.model import ModifiableModel
from sparseml.modifiers.logarithmic_equalization.pytorch import (
    LogarithmicEqualizationModifierPyTorch,
)
from tests.sparseml.pytorch.helpers import LinearNet


@pytest.mark.unit
class TestLogEqualizationMapping(unittest.TestCase):
    def setUp(self):
        self.model = ModifiableModel(framework=Framework.pytorch, model=LinearNet())
        self.state = State(framework=Framework.pytorch, model=self.model)

    def test_successful_map(self):
        mappings = [(["seq.fc2"], "seq.block1.fc1")]
        modifier = LogarithmicEqualizationModifierPyTorch(mappings=mappings)

        modifier.ignore = []
        modifier.resolved_mappings_ = modifier._resolve_mappings(self.state.model)

        self.assertEqual(len(modifier.resolved_mappings_), len(mappings))

        mapping = modifier.resolved_mappings_[0]
        self.assertEqual(mapping.smooth_name, mappings[0][1])
        self.assertIsInstance(mapping.smooth_layer, Linear)
        self.assertIsInstance(mapping.balance_layers[0], Linear)
