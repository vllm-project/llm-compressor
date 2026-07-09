import pytest

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning.wanda.base import WandaPruningModifier


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_wanda_is_registered():
    sparsity = 0.5
    targets = "__ALL_PRUNABLE__"
    type_ = ModifierFactory.create(
        type_="WandaPruningModifier",
        allow_experimental=False,
        allow_registered=True,
        sparsity=sparsity,
        targets=targets,
    )

    assert isinstance(
        type_, WandaPruningModifier
    ), "PyTorch WandaPruningModifier not registered"
