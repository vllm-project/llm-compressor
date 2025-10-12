import pytest
import torch

from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from tests.llmcompressor.modifiers.conf import LifecyleTestingHarness
from tests.llmcompressor.pytorch.helpers import LinearNet


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


@pytest.mark.unit
def test_create_default_quant_modifier():
    modifier = GPTQModifier(block_size=128, targets=["Linear"], scheme="FP8")

    testing_harness = LifecyleTestingHarness(model=LinearNet(), start=-1)
    modifier.initialize(testing_harness.get_state())
    modifier.on_start(testing_harness.get_state(), None)

    model = testing_harness.state.model
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            assert hasattr(module, "quantization_scheme")
            assert hasattr(module, "input_observer")
            assert hasattr(module, "weight_observer")
            pre_hooks = list(module._forward_pre_hooks.values())
            post_hooks = list(module._forward_hooks.values())
            assert pre_hooks[0].__name__ == "calibrate_input_hook"
            assert post_hooks[0].__name__ == "calibrate_module"


def _check_config(actual, expected):
    assert actual["num_bits"] == expected["num_bits"]
    assert actual["symmetric"] == expected["symmetric"]
    assert actual["strategy"] == expected["strategy"]


def test_set_quant_in_gptq():
    quant_kwargs = {
        "config_groups": {
            "config_group_0": {
                "targets": ["Linear"],
                "input_activations": {
                    "num_bits": 8,
                    "symmetric": False,
                    "strategy": "token",
                    "dynamic": True,
                },
                "weights": {
                    "num_bits": 4,
                    "symmetric": True,
                    "strategy": "channel",
                },
            }
        }
    }

    modifier = GPTQModifier(block_size=128, **quant_kwargs)
    config = modifier.resolve_quantization_config()

    _check_config(
        dict(config.config_groups["config_group_0"].weights),
        quant_kwargs["config_groups"]["config_group_0"]["weights"],
    )
    _check_config(
        dict(config.config_groups["config_group_0"].input_activations),
        quant_kwargs["config_groups"]["config_group_0"]["input_activations"],
    )
