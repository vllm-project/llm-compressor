import pytest
import torch

from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier


@pytest.mark.unit
@pytest.mark.parametrize("modifier_cls", [SparseGPTModifier, WandaPruningModifier])
def test_sequential_start_registers_hooks_only_in_scope(modifier_cls):
    model = torch.nn.ModuleDict(
        {
            "layers": torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(4, 4)) for _ in range(2)]
            )
        }
    )
    modifier = modifier_cls(sparsity=0.5)
    modifier._target_layers = {
        f"layers.{index}": layer for index, layer in enumerate(model.layers)
    }
    modifier._lazy_initialization = True

    modifier.on_sequential_epoch_start(
        state=None,
        event=None,
        modules=list(model.layers[0].modules()),
    )

    assert model.layers[0][0] in modifier._module_names
    assert model.layers[1][0] not in modifier._module_names
    assert len(modifier._hooks) == 1


@pytest.mark.unit
def test_owl_keeps_eager_initialization():
    modifier = SparseGPTModifier(
        sparsity=0.5,
        sparsity_profile="owl",
        owl_m=1,
        owl_lmbda=1.0,
    )

    assert not modifier.supports_lazy_initialization
