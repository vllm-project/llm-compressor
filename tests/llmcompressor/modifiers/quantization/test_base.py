import pytest
from compressed_tensors.quantization import ActivationOrdering

from llmcompressor.modifiers.quantization import GPTQModifier


@pytest.fixture
def q_config_kwargs(recipe_actorder_0, recipe_actorder_1):
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {
                    "num_bits": 8,
                    "symmetric": False,
                    "strategy": "token",
                    "dynamic": "true",
                    "kwargs": {},
                },
                "weights": {
                    "num_bits": 4,
                    "symmetric": True,
                    "strategy": "channel",
                    "kwargs": {},
                    "actorder": recipe_actorder_0,
                },
            },
            "group_1": {
                "targets": ["Linear"],
                "input_activations": {
                    "num_bits": 8,
                    "symmetric": False,
                    "strategy": "token",
                    "dynamic": "true",
                    "kwargs": {},
                },
                "weights": {
                    "num_bits": 4,
                    "symmetric": True,
                    "strategy": "channel",
                    "kwargs": {},
                    "actorder": recipe_actorder_1,
                },
            },
        },
    }


@pytest.mark.parametrize(
    "has_actorder,actorder,recipe_actorder_0,recipe_actorder_1,",
    [
        (True, "group", None),
        (True, "static"),
        (True, None),
        (True, ActivationOrdering.DYNAMIC),
        (True, ActivationOrdering.WEIGHT),
        (False, None),
    ],
)
def test_actorder_resolution(
    has_actorder, actorder, recipe_actorder_0, recipe_actorder_1, q_config_kwargs
):
    if has_actorder:
        modifier = GPTQModifier(**q_config_kwargs, actorder=actorder)
    else:
        modifier = GPTQModifier(**q_config_kwargs)

    config = modifier.resolve_quantization_config()
    assert config.config_groups["group_0"].input_activations.actorder is None
    assert config.config_groups["group_0"].weights.actorder == actorder
