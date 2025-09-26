from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.core.state import State
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier


@pytest.fixture
def model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return AutoModelForCausalLM.from_pretrained(
        "nm-testing/tinysmokellama-3.2", device_map=device
    )


@pytest.fixture
def dataloader():
    dataset = MagicMock()
    dataset.column_names = []
    dataloader = MagicMock()
    dataloader.dataset = dataset
    dataloader.__iter__.return_value = iter([])
    return dataloader


@pytest.mark.integration
@pytest.mark.parametrize("extra_targets,expected", [([], 0), (["lm_head"], 1)])
def test_lm_head(extra_targets, expected, model, dataloader):
    kwargs = {
        "sparsity": 0.5,
        "block_size": 128,
        "targets": [
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
            "model.layers.4",
            "model.layers.5",
        ]
        + extra_targets,
    }
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    modifier = SparseGPTModifier(**kwargs)

    state = State()
    state.update(model=model, device=device, calib_data=dataloader)
    modifier.initialize(state)
    modifier.on_start(state, None)

    assert len(model.lm_head._forward_hooks) == expected

    modifier.finalize(state)
