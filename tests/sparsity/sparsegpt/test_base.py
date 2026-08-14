import pytest

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from tests.llmcompressor.modifiers.conf import LifecyleTestingHarness
from tests.llmcompressor.pytorch.helpers import LinearNet


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_sparse_gpt_is_registered():
    sparsity = 0.5
    targets = "__ALL_PRUNABLE__"
    type_ = ModifierFactory.create(
        type_="SparseGPTModifier",
        allow_experimental=False,
        allow_registered=True,
        sparsity=sparsity,
        targets=targets,
    )

    assert isinstance(
        type_, SparseGPTModifier
    ), "PyTorch SparseGPTModifier not registered"


@pytest.mark.unit
@pytest.mark.parametrize(
    "sparsity, targets",
    [
        [[0.5, 0.2], "__ALL__"],
        [[0.2, 0.1, 0.3], ["seq.fc1", "seq.fc2"]],
        [[0.3, 0.4], ["re:.*fc1", "re:.*fc2"]],
    ],
)
def test_invalid_layerwise_recipes_raise_exceptions(sparsity, targets):
    modifier = SparseGPTModifier(
        sparsity=sparsity,
        block_size=128,
        targets=targets,
    )
    testing_harness = LifecyleTestingHarness(model=LinearNet(), start=-1)

    # confirm invalid layerwise recipes fail at initialization
    with pytest.raises(ValueError):
        modifier.initialize(testing_harness.get_state())


@pytest.mark.unit
def test_successful_layerwise_recipe():
    sparsities = [0.5, 0.2]
    targets = ["seq.fc1", "seq.fc2"]
    modifier = SparseGPTModifier(sparsity=sparsities, block_size=128, targets=targets)
    testing_harness = LifecyleTestingHarness(model=LinearNet(), start=-1)
    modifier.initialize(testing_harness.get_state())
    modifier.on_start(testing_harness.get_state(), None)

    model = testing_harness.state.model
    num_hooks = len(modifier._hooks)
    num_found = sum(len(module._forward_hooks) > 0 for module in model.modules())
    assert num_hooks == num_found


def _check_config(actual, expected):
    assert actual["num_bits"] == expected["num_bits"]
    assert actual["symmetric"] == expected["symmetric"]
    assert actual["strategy"] == expected["strategy"]
