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

from llmcompressor.modifiers.factory import ModifierFactory
from tests.llmcompressor.modifiers.conf import setup_modifier_factory
from tests.testing_utils import requires_torch


@pytest.mark.unit
@requires_torch
class TestWandaPytorchIsRegistered(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            sparsity=0.5,
            targets="__ALL_PRUNABLE__",
        )
        setup_modifier_factory()

    def test_wanda_pytorch_is_registered(self):
        from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier

        type_ = ModifierFactory.create(
            type_="WandaPruningModifier",
            allow_experimental=False,
            allow_registered=True,
            **self.kwargs,
        )

        self.assertIsInstance(
            type_,
            WandaPruningModifier,
            "PyTorch ConstantPruningModifier not registered",
        )
