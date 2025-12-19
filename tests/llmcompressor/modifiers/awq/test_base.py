from itertools import product

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)
from pydantic import ValidationError
from torch.nn import Linear
from torch.testing import assert_close

from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
from llmcompressor.modifiers.awq.base import (
    get_lowest_common_ancestor_with_avoid,
)
from llmcompressor.modifiers.factory import ModifierFactory


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_awq_is_registered():
    """Ensure AWQModifier is registered in ModifierFactory"""
    modifier = ModifierFactory.create(
        type_="AWQModifier",
        allow_experimental=False,
        allow_registered=True,
        scheme="W4A16_ASYM",
    )

    assert isinstance(modifier, AWQModifier), "AWQModifier not registered"


@pytest.mark.unit
def test_set_resolved_mappings():
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            ),
            AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
            AWQMapping(
                "re:.*up_proj",
                ["re:.*down_proj"],
            ),
        ],
        scheme="W4A16_ASYM",
    )
    self_attn = torch.nn.ModuleDict(
        {
            "q_proj": Linear(4, 4),
            "k_proj": Linear(4, 4),
            "v_proj": Linear(4, 4),
            "o_proj": Linear(4, 4),
        }
    )
    mlp = torch.nn.ModuleDict(
        {
            "experts": torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "gate_proj": Linear(4, 2),
                            "up_proj": Linear(4, 2),
                            "down_proj": Linear(2, 4),
                        }
                    )
                    for _ in range(3)
                ]
            )
        }
    )
    model = torch.nn.ModuleDict(
        {
            "decoder": torch.nn.ModuleDict(
                {
                    "self_attn": self_attn,
                    "input_layernorm": torch.nn.LayerNorm(4),
                    "mlp": mlp,
                }
            )
        }
    )
    awq._set_resolved_mappings(model)
    for mapping in awq._resolved_mappings:
        if "input_layernorm" in mapping.smooth_name:
            assert set(mapping.balance_names) == {
                "decoder.self_attn.q_proj",
                "decoder.self_attn.k_proj",
                "decoder.self_attn.v_proj",
            }
            assert set(mapping.balance_layers) == {
                self_attn.q_proj,
                self_attn.k_proj,
                self_attn.v_proj,
            }
            assert mapping.parent_name == "decoder.self_attn"
            assert mapping.parent == self_attn
        if "self_attn.v_proj" in mapping.smooth_name:
            assert set(mapping.balance_names) == {"decoder.self_attn.o_proj"}
            assert mapping.parent_name == "decoder.self_attn.o_proj"
        if "mlp.experts" in mapping.smooth_name and "up_proj" in mapping.smooth_name:
            expert_idx = mapping.smooth_name.split(".")[-2]
            expected_down_proj = f"decoder.mlp.experts.{expert_idx}.down_proj"
            assert set(mapping.balance_names) == {expected_down_proj}
            assert mapping.parent_name == expected_down_proj
            assert mapping.parent == mlp["experts"][int(expert_idx)]["down_proj"]

    awq = AWQModifier(
        mappings=[
            # make sure we exclude case where o_proj/v_proj shapes are mismatched
            AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
            # make sure we exclude mapping if any balance layers are skipped
            AWQMapping("re:.*v_proj", ["re:.*z_proj", "re:.*o_proj"]),
        ],
        scheme="W4A16_ASYM",
    )
    model = torch.nn.ModuleDict(
        {
            "decoder": torch.nn.ModuleDict(
                {
                    "self_attn": torch.nn.ModuleDict(
                        {
                            "q_proj": Linear(4, 2),
                            "k_proj": Linear(4, 2),
                            "v_proj": Linear(4, 2),
                            "z_proj": Linear(2, 4),
                            "o_proj": Linear(4, 4),
                        }
                    )
                }
            )
        }
    )
    awq._set_resolved_mappings(model)
    if len(awq._resolved_mappings) > 0:
        assert all(
            "o_proj" not in name for name in awq._resolved_mappings[0].balance_names
        ), "should have skipped v->o mapping because o is incompatible"
        assert all(
            "z_proj" not in name for name in awq._resolved_mappings[0].balance_names
        ), (
            "should have skipped v->[z,o] mapping because o is incompatible even though"
            "z is compatible"
        )
    assert len(awq._resolved_mappings) == 0


@pytest.mark.unit
def test_validate():
    AWQModifier(scheme="W4A16", duo_scaling="both")
    with pytest.raises(ValidationError):
        AWQModifier(scheme="W4A16", duo_scaling="Both")
    with pytest.raises(ValidationError):
        AWQModifier(scheme="W4A16", duo_scaling="x")


@pytest.mark.unit
def test_moe_multiple_balance_layers():
    """Test AWQ mapping with multiple balance layers in MoE architecture"""
    awq = AWQModifier(
        mappings=[
            # Map input_layernorm to multiple experts' gate_proj and up_proj
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*gate_proj", "re:.*up_proj"],
            ),
        ],
        scheme="W4A16_ASYM",
    )

    # Create a simplified MoE model structure
    mlp = torch.nn.ModuleDict(
        {
            "experts": torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "gate_proj": Linear(4, 4),
                            "up_proj": Linear(4, 4),
                            "down_proj": Linear(4, 4),
                        }
                    )
                    for _ in range(2)
                ]
            )
        }
    )
    model = torch.nn.ModuleDict(
        {
            "layer": torch.nn.ModuleDict(
                {
                    "input_layernorm": torch.nn.LayerNorm(4),
                    "mlp": mlp,
                }
            )
        }
    )

    awq._set_resolved_mappings(model)

    # Should have one mapping for input_layernorm
    assert len(awq._resolved_mappings) == 1
    mapping = awq._resolved_mappings[0]

    # Should map to all gate_proj and up_proj across all experts
    expected_balance_names = {
        "layer.mlp.experts.0.gate_proj",
        "layer.mlp.experts.0.up_proj",
        "layer.mlp.experts.1.gate_proj",
        "layer.mlp.experts.1.up_proj",
    }
    assert set(mapping.balance_names) == expected_balance_names

    parent_name, parent = get_lowest_common_ancestor_with_avoid(
        ["embed_tokens", "decoder.self_attn.v_proj"], model
    )
    assert parent_name == "" and parent == model


def _auto_awq_normalize(layers: list[torch.nn.Module], group_size) -> torch.Tensor:
    """
    Original AutoAwq implementation (need to call .mean(0) to get normalized layer
    means
    """
    # [STEP 1]: Compute per-channel mean of normalised weights
    # All layer weights are concatted together
    weight = torch.cat([bl.weight for bl in layers], dim=0)
    orig_shape = weight.shape
    # The weights are reshaped to be organised by quantization group
    if group_size is not None:
        weight = weight.view(-1, group_size)
    # Calculates the relative magnitude of the weights within
    # each of the quantization groups, and rescales each group
    # individually so that each group has weights on a 0-1 scale.
    weight.abs_()
    weight.div_(weight.amax(dim=1, keepdim=True) + 1e-6)
    return weight.view(orig_shape)


@torch.no_grad
@pytest.mark.unit
@pytest.mark.parametrize(
    "n_balance_layers, group_size, n_input_features",
    [
        (5, -1, 32),  # channel
        (4, 10, 40),  # group
        (4, torch.inf, 40),  # tensor
    ],
)
def test_compute_layer_means(n_balance_layers, group_size, n_input_features):
    """
    Confirm our logic to compute duo_scaling layer means via a running tally
    matches the original memory-intensive AutoAWQ implementation, which concats
    all balance layers into a single tensor before reducing to mean
    Large models were prone to fail at this step.
    """
    balance_layers = [
        torch.nn.Linear(n_input_features, 10) for _ in range(n_balance_layers)
    ]
    group_size_arg = None
    match group_size:
        case -1:
            strategy = QuantizationStrategy.CHANNEL
            group_size = balance_layers[0].weight.shape[1]
        case torch.inf:
            strategy = QuantizationStrategy.TENSOR
            group_size = n_input_features * 10
        case _:
            strategy = QuantizationStrategy.GROUP
            group_size_arg = group_size

    for balance_layer in balance_layers:
        setattr(
            balance_layer,
            "quantization_scheme",
            QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    strategy=strategy,
                    group_size=group_size_arg,
                ),
            ),
        )

    auto_awq_means = _auto_awq_normalize(balance_layers, group_size).mean(0)

    llmc_awq_means = AWQModifier._compute_layer_means(balance_layers).to(
        auto_awq_means.dtype
    )

    assert_close(auto_awq_means, llmc_awq_means)


@pytest.mark.unit
@torch.no_grad
def test_compute_layer_means_does_not_modify_weights():
    """
    Test that _compute_layer_means does not modify the original layer weights.
    This is a regression test for a bug where in-place operations (abs_, div_)
    were modifying the original weights.
    """
    # Create test layers with known weight values
    n_layers = 3
    n_input_features = 16
    layers = [torch.nn.Linear(n_input_features, 8) for _ in range(n_layers)]

    # Set up quantization scheme for channel-wise quantization
    for layer in layers:
        setattr(
            layer,
            "quantization_scheme",
            QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    strategy=QuantizationStrategy.CHANNEL,
                ),
            ),
        )

    # Store copies of original weights before calling _compute_layer_means
    original_weights = [layer.weight.clone() for layer in layers]

    # Call _compute_layer_means which should NOT modify the original weights
    AWQModifier._compute_layer_means(layers)

    # Verify that the original weights remain unchanged
    for i, layer in enumerate(layers):
        assert_close(
            layer.weight,
            original_weights[i],
            msg=f"Layer {i} weight was modified by _compute_layer_means",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "rows, cols, block_height, block_width",
    [
        (
            32,
            256,
            4,
            8,
        ),
        (
            10,
            10,
            10,
            10,
        ),
        (
            512,
            256,
            128,
            128,
        ),
        (
            4,
            3,
            2,
            1,
        ),
    ],
)
@torch.no_grad
def test_block_strategy_compute_layer_means(rows, cols, block_height, block_width):
    """
    Confirm our logic to compute layer means works for BLOCK quantization
    """
    lin = torch.nn.Linear(cols, rows)
    setattr(
        lin,
        "quantization_scheme",
        QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                strategy=QuantizationStrategy.BLOCK,
                block_structure=[block_height, block_width],
            ),
        ),
    )
    # main
    llmc_awq_means = AWQModifier._compute_layer_means([lin])

    # ref
    num_heights = rows // block_height
    num_widths = cols // block_width

    ref_weight = torch.zeros_like(lin.weight)
    with torch.no_grad():
        for i, j in product(range(num_heights), range(num_widths)):
            block = lin.weight[
                i * block_height : (i + 1) * block_height,
                j * block_width : (j + 1) * block_width,
            ].abs()
            block = block / (block.max() + 1e-6)
            ref_weight[
                i * block_height : (i + 1) * block_height,
                j * block_width : (j + 1) * block_width,
            ] = block
    ref_means = ref_weight.sum(0, dtype=torch.float64) / ref_weight.size(0)

    # auto awq
    # we first reshape the weight such that it is effectively per-channel quantization
    # so that we can use the existing _auto_awq_normalize function
    orig_shape = lin.weight.shape
    q_args = lin.quantization_scheme.weights
    block_height, block_width = q_args.block_structure
    lin.weight.data = (  # (row, col)
        lin.weight.unflatten(0, (-1, block_height))  # = (num_H*block_H, num_W*block_W)
        .unflatten(-1, (-1, block_width))
        .transpose(1, 2)  # ↳ (num_H, num_W, block_H, block_W)
    )
    auto_awq_means = (
        _auto_awq_normalize([lin], block_height * block_width)
        .transpose(1, 2)
        .reshape(orig_shape)
        .mean(0)
        .to(llmc_awq_means.dtype)
    )

    # check
    assert_close(llmc_awq_means, ref_means, atol=1e-5, rtol=1e-5)
    assert_close(llmc_awq_means, auto_awq_means, atol=1e-5, rtol=1e-5)
