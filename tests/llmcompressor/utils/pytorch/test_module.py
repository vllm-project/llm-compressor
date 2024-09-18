import unittest

import torch.nn as nn

from llmcompressor.utils.pytorch import get_parent_by_name


class TestGetParentByName(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(
            nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10), nn.Softmax(dim=1)
        )

    def test_get_parent_by_name(self):
        # Test getting the parent of a non-existent layer
        with self.assertRaises(ValueError):
            get_parent_by_name("non_existent_layer", self.model)

        # Test getting the parent of the first layer
        name, parent = get_parent_by_name("0", self.model)
        self.assertEqual(parent, self.model)

        # Test getting the parent of a nested layer
        nested_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Sequential(nn.ReLU(), nn.Linear(20, 10)),
            nn.Softmax(dim=1),
        )
        name, parent = get_parent_by_name("1.1", nested_model)
        self.assertEqual(parent, nested_model[1])
        self.assertEqual(name, "1")
