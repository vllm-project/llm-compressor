import unittest

import pytest

from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.factory import ModifierFactory
from tests.llmcompressor.modifiers.conf import setup_modifier_factory


@pytest.mark.unit
class TestAWQIsRegistered(unittest.TestCase):
    def setUp(self):
        self.kwargs = {}
        setup_modifier_factory()

    def test_awq_is_registered(self):
        modifier = ModifierFactory.create(
            type_="AWQModifier",
            allow_experimental=False,
            allow_registered=True,
            **self.kwargs,
        )

        self.assertIsInstance(
            modifier,
            AWQModifier,
            "PyTorch AWQModifier not registered",
        )
