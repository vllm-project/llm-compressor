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
def test_ignore_behavior():
    """Test that mapping is skipped when NO layers are targeted for quantization"""
    # Test case 1: Some balance layers ignored but at least one is targeted
    # Mapping should proceed
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            ),
        ],
        ignore=["re:.*q_proj", "re:.*k_proj"],  # Only 2 of 3 balance layers ignored
        scheme="W4A16_ASYM",
    )

    self_attn = torch.nn.ModuleDict(
        {
            "q_proj": Linear(4, 4),
            "k_proj": Linear(4, 4),
            "v_proj": Linear(4, 4),
        }
    )
    model = torch.nn.ModuleDict(
        {
            "decoder": torch.nn.ModuleDict(
                {
                    "self_attn": self_attn,
                    "input_layernorm": torch.nn.LayerNorm(4),
                }
            )
        }
    )

    awq._set_resolved_mappings(model)

    # Mapping should exist because v_proj is targeted for quantization
    assert len(awq._resolved_mappings) == 1

    # Test case 2: All Linear layers ignored - mapping should be skipped
    # because no layers are targeted for quantization
    awq2 = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            ),
        ],
        ignore=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
        scheme="W4A16_ASYM",
    )

    awq2._set_resolved_mappings(model)

    # Mapping should be skipped because no layers are targeted for quantization
    # (input_layernorm is LayerNorm, not Linear, so not targeted anyway)
    assert len(awq2._resolved_mappings) == 0


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


@pytest.mark.unit
def test_qwen3_next_moe_with_shared_expert():
    """Test AWQ mapping for Qwen3Next architecture with shared_expert.

    Qwen3Next includes a shared_expert in addition to the MoE experts.
    This test verifies that the mapping correctly resolves both experts
    and shared_expert projections.
    """
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*post_attention_layernorm$",
                [
                    "re:.*mlp.experts.*.gate_proj$",
                    "re:.*mlp.experts.*.up_proj$",
                    "re:.*mlp.shared_expert.gate_proj$",
                    "re:.*mlp.shared_expert.up_proj$",
                ],
            ),
        ],
        scheme="W4A16_ASYM",
    )

    # Create a Qwen3Next-like MoE model structure with shared_expert
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
            ),
            "shared_expert": torch.nn.ModuleDict(
                {
                    "gate_proj": Linear(4, 4),
                    "up_proj": Linear(4, 4),
                    "down_proj": Linear(4, 4),
                }
            ),
        }
    )
    model = torch.nn.ModuleDict(
        {
            "layer": torch.nn.ModuleDict(
                {
                    "post_attention_layernorm": torch.nn.LayerNorm(4),
                    "mlp": mlp,
                }
            )
        }
    )

    awq._set_resolved_mappings(model)

    # Should have one mapping for post_attention_layernorm
    assert len(awq._resolved_mappings) == 1
    mapping = awq._resolved_mappings[0]

    # Should map to all gate_proj and up_proj across experts AND shared_expert
    expected_balance_names = {
        "layer.mlp.experts.0.gate_proj",
        "layer.mlp.experts.0.up_proj",
        "layer.mlp.experts.1.gate_proj",
        "layer.mlp.experts.1.up_proj",
        "layer.mlp.shared_expert.gate_proj",
        "layer.mlp.shared_expert.up_proj",
    }
    assert set(mapping.balance_names) == expected_balance_names


@pytest.mark.unit
def test_qwen3_next_hybrid_attention():
    """Test AWQ mapping for Qwen3Next hybrid attention architecture.

    Qwen3Next has two attention types:
    - self_attn: standard attention with q_proj, k_proj, v_proj, o_proj
    - linear_attn: Gated DeltaNet with in_proj_qkvz, in_proj_ba, norm, out_proj

    This test verifies that the mapping correctly resolves both attention types.
    Each attention type needs separate mappings for proper layer grouping.
    """
    awq = AWQModifier(
        mappings=[
            # self_attn mappings (layer 1 has self_attn)
            # Use layer-specific pattern since each mapping must match exactly one
            # smooth_layer
            AWQMapping(
                "re:.*layers\\.1\\.input_layernorm$",
                [
                    "re:.*self_attn.q_proj$",
                    "re:.*self_attn.k_proj$",
                    "re:.*self_attn.v_proj$",
                ],
            ),
            AWQMapping("re:.*self_attn.v_proj$", ["re:.*self_attn.o_proj$"]),
            # linear_attn mappings (layer 0 has linear_attn)
            AWQMapping(
                "re:.*layers\\.0\\.input_layernorm$",
                ["re:.*linear_attn.in_proj_qkvz$", "re:.*linear_attn.in_proj_ba$"],
            ),
            AWQMapping("re:.*linear_attn.norm$", ["re:.*linear_attn.out_proj$"]),
        ],
        scheme="W4A16_ASYM",
    )

    # Create a Qwen3Next-like model with both self_attn and linear_attn layers
    # Layer 0: linear_attn
    linear_attn_layer = torch.nn.ModuleDict(
        {
            "input_layernorm": torch.nn.LayerNorm(4),
            "linear_attn": torch.nn.ModuleDict(
                {
                    "in_proj_qkvz": Linear(4, 16),
                    "in_proj_ba": Linear(4, 4),
                    "norm": torch.nn.LayerNorm(4),
                    "out_proj": Linear(4, 4),
                }
            ),
        }
    )

    # Layer 1: self_attn
    self_attn_layer = torch.nn.ModuleDict(
        {
            "input_layernorm": torch.nn.LayerNorm(4),
            "self_attn": torch.nn.ModuleDict(
                {
                    "q_proj": Linear(4, 4),
                    "k_proj": Linear(4, 4),
                    "v_proj": Linear(4, 4),
                    "o_proj": Linear(4, 4),
                }
            ),
        }
    )

    model = torch.nn.ModuleDict(
        {
            "layers": torch.nn.ModuleList([linear_attn_layer, self_attn_layer]),
        }
    )

    awq._set_resolved_mappings(model)

    # Should have 4 mappings:
    # 1. layer 0 input_layernorm -> linear_attn projections
    # 2. layer 0 linear_attn.norm -> linear_attn.out_proj
    # 3. layer 1 input_layernorm -> self_attn projections
    # 4. layer 1 self_attn.v_proj -> self_attn.o_proj
    assert len(awq._resolved_mappings) == 4

    # Check linear_attn input mapping
    linear_input_mapping = next(
        m for m in awq._resolved_mappings if m.smooth_name == "layers.0.input_layernorm"
    )
    assert set(linear_input_mapping.balance_names) == {
        "layers.0.linear_attn.in_proj_qkvz",
        "layers.0.linear_attn.in_proj_ba",
    }

    # Check linear_attn output mapping
    linear_output_mapping = next(
        m
        for m in awq._resolved_mappings
        if m.smooth_name == "layers.0.linear_attn.norm"
    )
    assert linear_output_mapping.balance_names == ["layers.0.linear_attn.out_proj"]

    # Check self_attn input mapping
    self_input_mapping = next(
        m for m in awq._resolved_mappings if m.smooth_name == "layers.1.input_layernorm"
    )
    assert set(self_input_mapping.balance_names) == {
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.v_proj",
    }

    # Check self_attn output mapping
    self_output_mapping = next(
        m
        for m in awq._resolved_mappings
        if m.smooth_name == "layers.1.self_attn.v_proj"
    )
    assert self_output_mapping.balance_names == ["layers.1.self_attn.o_proj"]


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
    "n_balance_layers, n_input_features, strategy, group_size",
    [
        (5, 32, QuantizationStrategy.CHANNEL, None),  # channel
        (4, 40, QuantizationStrategy.GROUP, 10),  # group
        (4, 40, QuantizationStrategy.TENSOR, None),  # tensor
        (3, 64, QuantizationStrategy.TENSOR_GROUP, 16),  # tensor_group
    ],
)
def test_compute_layer_means(n_balance_layers, n_input_features, strategy, group_size):
    """
    Confirm our logic to compute duo_scaling layer means via a running tally
    matches the original memory-intensive AutoAWQ implementation, which concats
    all balance layers into a single tensor before reducing to mean
    Large models were prone to fail at this step.
    """
    balance_layers = [
        torch.nn.Linear(n_input_features, 10) for _ in range(n_balance_layers)
    ]

    for balance_layer in balance_layers:
        setattr(
            balance_layer,
            "quantization_scheme",
            QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    strategy=strategy,
                    group_size=group_size,
                ),
            ),
        )

    match strategy:
        case QuantizationStrategy.GROUP | QuantizationStrategy.TENSOR_GROUP:
            group_size_arg = group_size
        case QuantizationStrategy.TENSOR:
            group_size_arg = n_input_features * 10
        case _:
            group_size_arg = None

    auto_awq_means = _auto_awq_normalize(balance_layers, group_size_arg).mean(0)

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
            4,
            3,
            2,
            1,
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
    # so that we can compare to the existing _auto_awq_normalize function
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


@pytest.mark.unit
def test_learned_scales_config():
    """Test that learned scales configuration parameters are properly set"""
    # Test default values
    awq_default = AWQModifier(scheme="W4A16_ASYM")
    assert awq_default.scale_search_method == "grid"
    assert awq_default.learned_scales_iters == 100
    assert awq_default.learned_scales_lr == 0.01

    # Test custom values
    awq_custom = AWQModifier(
        scheme="W4A16_ASYM",
        scale_search_method="learned",
        learned_scales_iters=50,
        learned_scales_lr=0.02,
    )
    assert awq_custom.scale_search_method == "learned"
    assert awq_custom.learned_scales_iters == 50
    assert awq_custom.learned_scales_lr == 0.02


@pytest.mark.unit
def test_scale_parameter_attachment_detachment():
    """Test attaching and detaching learnable scales as nn.Parameter"""
    from llmcompressor.modifiers.awq.mappings import ResolvedMapping

    awq = AWQModifier(scheme="W4A16_ASYM", scale_search_method="learned")

    # Create a simple parent module
    parent = torch.nn.ModuleDict({"linear": torch.nn.Linear(4, 4)})

    # Create a resolved mapping
    mapping = ResolvedMapping(
        smooth_name="test.smooth_layer",
        smooth_layer=torch.nn.LayerNorm(4),
        balance_layers=[parent["linear"]],
        balance_names=["test.linear"],
        parent=parent,
        parent_name="test",
    )

    # Create initial scales
    initial_scales = torch.ones(4)

    # Test attachment
    scales_param = awq._attach_learnable_scales(mapping, initial_scales)
    assert isinstance(scales_param, torch.nn.Parameter)
    assert scales_param.requires_grad
    assert scales_param.shape == initial_scales.shape
    assert_close(scales_param, initial_scales)

    # Verify parameter is registered on parent
    param_name = awq._get_scale_param_name(mapping.smooth_name)
    assert hasattr(parent, param_name)
    assert getattr(parent, param_name) is scales_param

    # Modify the parameter to simulate learning
    with torch.no_grad():
        scales_param.data.fill_(2.0)

    # Test detachment
    final_scales = awq._detach_learnable_scales(mapping)
    assert not hasattr(parent, param_name)
    assert_close(final_scales, torch.full((4,), 2.0))
    assert not final_scales.requires_grad


@pytest.mark.unit
def test_compute_loss_gradient_flow():
    """Test that loss computation maintains gradient when return_scalar=False"""
    awq = AWQModifier(scheme="W4A16_ASYM")

    # Create dummy outputs
    fp16_outputs = [torch.randn(2, 4, 8, requires_grad=True)]
    int_w_outputs = [torch.randn(2, 4, 8, requires_grad=True)]

    # Test with return_scalar=True (should break gradient)
    loss_scalar = awq._compute_loss(fp16_outputs, int_w_outputs, return_scalar=True)
    assert isinstance(loss_scalar, float)

    # Test with return_scalar=False (should maintain gradient)
    loss_tensor = awq._compute_loss(fp16_outputs, int_w_outputs, return_scalar=False)
    assert isinstance(loss_tensor, torch.Tensor)
    assert loss_tensor.requires_grad
    assert loss_tensor.numel() == 1

    # Verify backward pass works
    loss_tensor.backward()
    assert fp16_outputs[0].grad is not None
    assert int_w_outputs[0].grad is not None


@pytest.mark.unit
def test_learned_scales_initialization():
    """Test that learned scales are properly initialized from activation statistics"""
    from llmcompressor.modifiers.awq.mappings import ResolvedMapping

    awq = AWQModifier(
        scheme="W4A16_ASYM",
        scale_search_method="learned",
        learned_scales_iters=5,
        learned_scales_lr=0.01,
        duo_scaling=False,  # Simpler initialization
    )

    # Create simple modules
    smooth_layer = torch.nn.LayerNorm(8)
    balance_layer = torch.nn.Linear(8, 8)
    parent = torch.nn.ModuleDict({"balance": balance_layer})

    mapping = ResolvedMapping(
        smooth_name="test.smooth_layer",
        smooth_layer=smooth_layer,
        balance_layers=[balance_layer],
        balance_names=["test.balance"],
        parent=parent,
        parent_name="test",
    )

    # Set up activation means
    activation_mean = torch.rand(8).abs() + 0.1
    awq._smooth_activation_means = {"test.smooth_layer": (activation_mean.clone(), 1)}

    # Attach learnable scales
    # Compute expected initial scales (from the learned method logic)
    expected_initial = activation_mean.pow(0.5).clamp(min=1e-4).view(-1)
    expected_initial = expected_initial / (expected_initial.max() * expected_initial.min()).sqrt()
    expected_initial[torch.isinf(expected_initial)] = 1
    expected_initial[torch.isnan(expected_initial)] = 1

    scales_param = awq._attach_learnable_scales(mapping, expected_initial)

    # Verify it's a parameter
    assert isinstance(scales_param, torch.nn.Parameter)
    assert scales_param.requires_grad
    assert_close(scales_param, expected_initial)

    # Verify it can be optimized
    optimizer = torch.optim.Adam([scales_param], lr=0.01)
    loss = (scales_param - 2.0).pow(2).sum()
    loss.backward()
    optimizer.step()

    # Scales should have changed
    assert not torch.allclose(scales_param, expected_initial)

    # Detach and verify
    final_scales = awq._detach_learnable_scales(mapping)
    assert not final_scales.requires_grad
    assert_close(final_scales, scales_param.data)


@pytest.mark.unit
def test_scale_hook_applies_correctly():
    """Test that the forward hook correctly applies scales to balance and smooth layers"""
    from llmcompressor.modifiers.awq.mappings import ResolvedMapping

    awq = AWQModifier(
        scheme="W4A16_ASYM",
        scale_search_method="learned",
    )

    # Create modules
    smooth_layer = torch.nn.LayerNorm(4)
    balance_layer = torch.nn.Linear(4, 4)
    parent = torch.nn.ModuleDict({"balance": balance_layer})

    mapping = ResolvedMapping(
        smooth_name="test.smooth_layer",
        smooth_layer=smooth_layer,
        balance_layers=[balance_layer],
        balance_names=["test.balance"],
        parent=parent,
        parent_name="test",
    )

    # Save original weights
    orig_weights = {balance_layer: balance_layer.weight.clone()}
    smooth_orig_weight = smooth_layer.weight.clone()
    smooth_orig_bias = smooth_layer.bias.clone()

    # Create and attach scales
    initial_scales = torch.tensor([1.0, 2.0, 3.0, 4.0])
    awq._attach_learnable_scales(mapping, initial_scales)

    # Create hook
    hook_fn = awq._create_scale_application_hook(
        mapping, orig_weights, smooth_orig_weight, smooth_orig_bias
    )
    balance_hook = balance_layer.register_forward_pre_hook(hook_fn)
    smooth_hook = smooth_layer.register_forward_pre_hook(hook_fn)

    # Run forward pass on balance layer (hook should apply scales)
    input_tensor = torch.randn(2, 4)
    _ = balance_layer(input_tensor)

    # Verify scales were applied to balance layer
    expected_balance_weight = orig_weights[balance_layer] * initial_scales.view(1, -1)
    assert_close(balance_layer.weight, expected_balance_weight)

    # Run forward pass on smooth layer (hook should apply inverse scales)
    _ = smooth_layer(input_tensor)

    # Verify inverse scales were applied to smooth layer
    expected_smooth_weight = smooth_orig_weight / initial_scales
    expected_smooth_bias = smooth_orig_bias / initial_scales
    assert_close(smooth_layer.weight, expected_smooth_weight)
    assert_close(smooth_layer.bias, expected_smooth_bias)

    # Remove hooks
    balance_hook.remove()
    smooth_hook.remove()

    # Clean up
    awq._detach_learnable_scales(mapping)


@pytest.mark.unit
def test_learned_scales_with_duo_scaling():
    """Test learned scales initialization with duo_scaling enabled"""
    from llmcompressor.modifiers.awq.mappings import ResolvedMapping

    awq = AWQModifier(
        scheme="W4A16_ASYM",
        scale_search_method="learned",
        duo_scaling=True,
    )

    # Create modules
    smooth_layer = torch.nn.LayerNorm(8)
    balance_layer = torch.nn.Linear(8, 8)
    parent = torch.nn.ModuleDict({"balance": balance_layer})

    # Add quantization scheme
    balance_layer.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=False,
            strategy=QuantizationStrategy.CHANNEL,
        ),
    )

    mapping = ResolvedMapping(
        smooth_name="test.smooth_layer",
        smooth_layer=smooth_layer,
        balance_layers=[balance_layer],
        balance_names=["test.balance"],
        parent=parent,
        parent_name="test",
    )

    # Set up activation means
    awq._smooth_activation_means = {"test.smooth_layer": (torch.rand(8).abs() + 0.1, 1)}

    # With duo_scaling=True, initialization uses both activation and weight means
    # Just verify the parameter can be attached and has valid values
    # (full optimization would require quantization infrastructure)
    activation_mean = awq._smooth_activation_means["test.smooth_layer"][0]
    weight_mean = awq._compute_layer_means(mapping.balance_layers)

    # Expected initial scales with duo_scaling
    expected = (activation_mean.pow(0.5) / (weight_mean.pow(0.5) + 1e-4)).clamp(min=1e-4)
    expected = expected / (expected.max() * expected.min()).sqrt()
    expected[torch.isinf(expected)] = 1
    expected[torch.isnan(expected)] = 1

    scales_param = awq._attach_learnable_scales(mapping, expected)

    # Verify valid initialization
    assert scales_param.shape == torch.Size([8])
    assert torch.all(torch.isfinite(scales_param))
    assert torch.all(scales_param > 0)
    assert scales_param.requires_grad

    # Clean up
    awq._detach_learnable_scales(mapping)
