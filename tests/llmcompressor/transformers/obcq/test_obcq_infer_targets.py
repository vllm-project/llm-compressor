import unittest

import pytest

from llmcompressor.utils.pytorch.module import get_no_split_params
from tests.testing_utils import requires_torch


@pytest.mark.integration
@requires_torch
class TestInferTargets(unittest.TestCase):
    def setUp(self):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M")
        self.modifiable_model = model
        self.targets = get_no_split_params(self.modifiable_model)

    def test_infer_targets(self):
        from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier

        self.assertEqual(len(self.targets), 1)
        self.assertEqual(self.targets[0], "LlamaDecoderLayer")

        modifier = SparseGPTModifier(sparsity=0.5)
        modifier.targets = self.targets
        modifier.model = self.modifiable_model
        compressible_layers = modifier.compressible_layers()
        self.assertEqual(len(compressible_layers), 6)
