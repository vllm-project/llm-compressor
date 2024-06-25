import unittest

import pytest

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.smoothquant.base import SmoothQuantModifier
from tests.llmcompressor.modifiers.conf import setup_modifier_factory


@pytest.mark.unit
class TestSmoothQuantIsRegistered(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            smoothing_strength=0.3,
            mappings=[(["layer1", "layer2"], "layer3")],
        )
        setup_modifier_factory()

    def test_smooth_quant_is_registered(self):
        modifier = ModifierFactory.create(
            type_="SmoothQuantModifier",
            allow_experimental=False,
            allow_registered=True,
            **self.kwargs,
        )

        self.assertIsInstance(
            modifier,
            SmoothQuantModifier,
            "PyTorch SmoothQuant not registered",
        )

        self.assertEqual(modifier.smoothing_strength, self.kwargs["smoothing_strength"])
        self.assertEqual(modifier.mappings, self.kwargs["mappings"])
