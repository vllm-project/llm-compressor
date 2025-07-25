from contextlib import nullcontext

import pytest

from llmcompressor.modifiers.quantization import GPTQModifier


@pytest.fixture
def q_config_kwargs(config_0, config_1):
    return dict(
        config_groups=dict(
            group_0=dict(
                targets=["Linear"],
                input_activations=dict(num_bits=8, symmetric=False, strategy="token"),
                weights=dict(
                    num_bits=4,
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                    actorder=config_0,
                ),
            ),
            group_1=dict(
                targets=["Linear"],
                input_activations=dict(num_bits=8, symmetric=False, strategy="token"),
                weights=dict(
                    num_bits=4,
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                    actorder=config_1,
                ),
            ),
        )
    )


@pytest.fixture
def block_q_config_kwargs():
    return dict(
        config_groups=dict(
            group_block=dict(
                targets=["Linear"],
                input_activations=dict(
                    num_bits=8, symmetric=True, strategy="group", group_size=128
                ),
                weights=dict(
                    num_bits=8,
                    symmetric=True,
                    strategy="block",
                    block_structure=[128, 128],
                ),
            ),
        )
    )


def test_block_strategy_parsing(block_q_config_kwargs):
    modifier = GPTQModifier(**block_q_config_kwargs)
    resolved = modifier.resolve_quantization_config()
    w_scheme = resolved.config_groups["group_block"].weights
    assert w_scheme.strategy == "block"
    assert w_scheme.block_structure == [128, 128]


@pytest.mark.parametrize(
    "has_actorder,actorder,config_0,config_1,expected_0,expected_1",
    [
        # defaults to "static" if nothing provided
        (False, "N/A", None, None, "static", "static"),
        # modifier overrides config if no config provided
        (True, "group", None, None, "group", "group"),
        # modifier overrides if config partially matches anyways
        (True, "group", None, "group", "group", "group"),
        (True, "group", "group", None, "group", "group"),
        # modifier errors if explicitly conflicts with config
        (True, "static", None, "group", "error", "error"),
        (True, "static", "group", None, "error", "error"),
        (True, "group", None, "static", "error", "error"),
        (True, "group", "static", None, "error", "error"),
        # modifier overrides to static if nothing is provided
        (False, "N/A", None, "static", "static", "static"),
        (False, "N/A", "static", None, "static", "static"),
        (False, "N/A", "static", "static", "static", "static"),
        # modifier does not override set config vaules
        (False, "N/A", None, "group", "static", "group"),
        (False, "N/A", "group", None, "group", "static"),
        (False, "N/A", "group", "group", "group", "group"),
    ],
)
def test_actorder_resolution(
    has_actorder, actorder, q_config_kwargs, expected_0, expected_1
):
    if has_actorder:
        modifier = GPTQModifier(**q_config_kwargs, actorder=actorder)
    else:
        modifier = GPTQModifier(**q_config_kwargs)

    with pytest.raises(ValueError) if expected_0 == "error" else nullcontext():
        resolved = modifier.resolve_quantization_config()

    if expected_0 != "error":
        assert resolved.config_groups["group_0"].input_activations.actorder is None
        assert resolved.config_groups["group_0"].weights.actorder == expected_0
        assert resolved.config_groups["group_1"].input_activations.actorder is None
        assert resolved.config_groups["group_1"].weights.actorder == expected_1
