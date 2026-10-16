from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor.core.state import State
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier


@pytest.fixture
def model():
    device = 0 if torch.accelerator.is_available() else "cpu"
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
    device = 0 if torch.accelerator.is_available() else "cpu"

    modifier = SparseGPTModifier(**kwargs)

    state = State()
    state.update(model=model, device=device, calib_data=dataloader)
    modifier.initialize(state)
    modifier.on_start(state, None)

    assert len(model.lm_head._forward_hooks) == expected

    modifier.finalize(state)


@pytest.mark.integration
@pytest.mark.parametrize("modifier_cls", [SparseGPTModifier, WandaPruningModifier])
@pytest.mark.parametrize(
    "ignore,ignored_name",
    [
        (["lm_head"], "lm_head"),
        (["re:.*lm_head$"], "lm_head"),
        (["model.layers.0.mlp.down_proj"], "model.layers.0.mlp.down_proj"),
    ],
)
def test_ignore_matches_module_names(
    modifier_cls, ignore, ignored_name, model, dataloader
):
    # with the default targets, each target layer is a single Linear module
    device = 0 if torch.accelerator.is_available() else "cpu"
    modifier = modifier_cls(sparsity=0.5, ignore=ignore)

    state = State()
    state.update(model=model, device=device, calib_data=dataloader)
    modifier.initialize(state)
    modifier.on_calibration_start(state, None)

    assert len(model.get_submodule(ignored_name)._forward_hooks) == 0
    assert len(model.model.layers[1].mlp.up_proj._forward_hooks) == 1

    modifier.finalize(state)
