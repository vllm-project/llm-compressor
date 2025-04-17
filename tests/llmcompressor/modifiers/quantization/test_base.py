import unittest

import pytest

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.quantization import QuantizationModifier
from tests.llmcompressor.modifiers.conf import setup_modifier_factory


@pytest.mark.unit
class TestQuantizationRegistered(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()
        self.kwargs = dict(
            index=0, group="quantization", start=2.0, end=-1.0, config_groups={}
        )

    def test_quantization_registered(self):
        quant_obj = ModifierFactory.create(
            type_="QuantizationModifier",
            allow_experimental=False,
            allow_registered=True,
            **self.kwargs,
        )

        self.assertIsInstance(quant_obj, QuantizationModifier)
