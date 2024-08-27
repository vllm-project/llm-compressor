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


@pytest.mark.unit
class TestSmoothQuantDefaults(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()

    def test_defaults(self):
        default_sq = SmoothQuantModifier()
        assert default_sq.smoothing_strength == 0.5

    def test_override_defaults(self):
        strength = 0.7
        dummy_map = [(["layer1", "layer2"], "layer3")]
        non_default_sq = SmoothQuantModifier(
            smoothing_strength=strength, mappings=dummy_map
        )

        assert non_default_sq.smoothing_strength == strength
        assert non_default_sq.mappings == dummy_map
