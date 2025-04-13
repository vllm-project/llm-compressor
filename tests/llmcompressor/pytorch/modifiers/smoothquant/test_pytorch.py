import unittest

import pytest
from torch.nn import Linear

from llmcompressor.core import State
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from tests.llmcompressor.pytorch.helpers import LinearNet


@pytest.mark.unit
class TestSmoothQuantMapping(unittest.TestCase):
    def setUp(self):
        self.model = LinearNet()
        self.state = State(model=self.model)

    def test_successful_map(self):
        mappings = [(["seq.fc1"], "seq.fc2")]
        modifier = SmoothQuantModifier(mappings=mappings)

        modifier.ignore = []
        modifier.resolved_mappings_ = modifier._resolve_mappings(self.state.model)

        self.assertEqual(len(modifier.resolved_mappings_), len(mappings))

        mapping = modifier.resolved_mappings_[0]
        self.assertEqual(mapping.smooth_name, mappings[0][1])
        self.assertIsInstance(mapping.smooth_layer, Linear)
        self.assertIsInstance(mapping.balance_layers[0], Linear)
