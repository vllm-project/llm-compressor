import pytest

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier
from tests.llmcompressor.modifiers.conf import setup_modifier_factory


@pytest.mark.unit
def test_wanda_pytorch_is_registered():
    setup_modifier_factory()

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
    ), "PyTorch ConstantPruningModifier not registered"
