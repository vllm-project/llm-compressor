import unittest

import pytest

from tests.testing_utils import requires_torch


@pytest.mark.integration
@requires_torch
class TestLMHead(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import AutoModelForCausalLM

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            "Xenova/llama2.c-stories15M", device_map=self.device
        )
        self.kwargs = {
            "sparsity": 0.5,
            "block_size": 128,
            "quantize": False,
            "targets": [
                "model.layers.0",
                "model.layers.1",
                "model.layers.2",
                "model.layers.3",
                "model.layers.4",
                "model.layers.5",
            ],
        }

    def test_lm_head_target(self):
        from llmcompressor.core.state import State
        from llmcompressor.modifiers.obcq import SparseGPTModifier

        sparsegpt_modifier_no_head = SparseGPTModifier(**self.kwargs)

        state = State()
        state.update(model=self.model, device=self.device)
        sparsegpt_modifier_no_head.initialize_compression(state.model)

        self.kwargs["targets"].append("lm_head")
        sparsegpt_modifier_head = SparseGPTModifier(**self.kwargs)
        sparsegpt_modifier_head.initialize_compression(state.model)

        # check we pick up the lm_head layer
        layers_no_head = len(sparsegpt_modifier_no_head.compressible_layers_)
        layers_head = len(sparsegpt_modifier_head.compressible_layers_)
        self.assertEqual(layers_head, layers_no_head + 1)
