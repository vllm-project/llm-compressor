import unittest
from unittest.mock import MagicMock

import pytest

from llmcompressor.core.state import State
from llmcompressor.modifiers.obcq import SparseGPTModifier


@pytest.mark.integration
class TestLMHead(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import AutoModelForCausalLM

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            "nm-testing/llama2.c-stories15M", device_map=self.device
        )

        self.kwargs = {
            "sparsity": 0.5,
            "block_size": 128,
            "targets": [
                "model.layers.0",
                "model.layers.1",
                "model.layers.2",
                "model.layers.3",
                "model.layers.4",
                "model.layers.5",
            ],
        }

        dataset = MagicMock()
        dataset.column_names = []
        self.dataloader = MagicMock()
        self.dataloader.dataset = dataset
        self.dataloader.__iter__.return_value = iter([])

    def test_no_lm_head_target(self):
        modifier = SparseGPTModifier(**self.kwargs)

        state = State()
        state.update(model=self.model, device=self.device, calib_data=self.dataloader)
        modifier.initialize(state)
        modifier.on_start(state, None)

        assert len(self.model.lm_head._forward_hooks) <= 0

        modifier.finalize(state)

    def test_lm_head_target(self):
        self.kwargs["targets"].append("lm_head")
        modifier = SparseGPTModifier(**self.kwargs)

        state = State()
        state.update(model=self.model, device=self.device, calib_data=self.dataloader)
        modifier.initialize(state)
        modifier.on_start(state, None)

        assert len(self.model.lm_head._forward_hooks) == 1

        modifier.finalize(state)
