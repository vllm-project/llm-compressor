import unittest

import pytest

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.logarithmic_equalization.base import (
    LogarithmicEqualizationModifier,
)
from llmcompressor.modifiers.smoothquant.base import SmoothQuantModifier
from tests.llmcompressor.modifiers.conf import setup_modifier_factory


@pytest.mark.unit
class TestLogarithmicEqualizationIsRegistered(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            smoothing_strength=0.3,
            mappings=[(["layer1", "layer2"], "layer3")],
        )
        setup_modifier_factory()

    def test_log_equalization_is_registered(self):
        modifier = ModifierFactory.create(
            type_="LogarithmicEqualizationModifier",
            allow_experimental=False,
            allow_registered=True,
            **self.kwargs,
        )

        self.assertIsInstance(
            modifier,
            LogarithmicEqualizationModifier,
            "PyTorch LogarithmicEqualizationModifier not registered",
        )

        self.assertIsInstance(modifier, SmoothQuantModifier)
        self.assertEqual(modifier.smoothing_strength, self.kwargs["smoothing_strength"])
        self.assertEqual(modifier.mappings, self.kwargs["mappings"])
