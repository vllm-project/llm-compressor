import unittest

import pytest

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.obcq.base import SparseGPTModifier
from tests.llmcompressor.modifiers.conf import setup_modifier_factory


@pytest.mark.unit
class TestSparseGPTIsRegistered(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            sparsity=0.5,
            targets="__ALL_PRUNABLE__",
        )
        setup_modifier_factory()

    def test_wanda_is_registered(self):
        type_ = ModifierFactory.create(
            type_="SparseGPTModifier",
            allow_experimental=False,
            allow_registered=True,
            **self.kwargs,
        )

        self.assertIsInstance(
            type_,
            SparseGPTModifier,
            "PyTorch SparseGPTModifier not registered",
        )
