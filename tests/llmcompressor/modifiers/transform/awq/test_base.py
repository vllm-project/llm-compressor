from itertools import product

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
    apply_quantization_config,
)
from datasets import Dataset
from pydantic import ValidationError
from torch.nn import Linear
from torch.testing import assert_close
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier
from llmcompressor.modifiers.transform.awq import base as awq_base
from llmcompressor.modifiers.transform.awq.base import (
    _compress_scales_for_gqa,
    _expand_scales_for_gqa,
    _get_head_dim,
    _is_gqa_projection,
    get_lowest_common_ancestor_with_avoid,
)
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.utils import get_high_precision
from tests.testing_utils import requires_compute_capability, requires_gpu


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_awq_is_registered():
    """Ensure AWQModifier is registered in ModifierFactory"""
    modifier = ModifierFactory.create(
        type_="AWQModifier",
        allow_experimental=False,
        allow_registered=True,
        duo_scaling="both",
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
    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
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
    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
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
    AWQModifier(duo_scaling="both")
    with pytest.raises(ValidationError):
        AWQModifier(duo_scaling="Both")
    with pytest.raises(ValidationError):
        AWQModifier(duo_scaling="x")


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
    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
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

    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
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

    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
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
    ref_means = ref_weight.sum(0, dtype=get_high_precision()) / ref_weight.size(0)

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


@requires_gpu(1)
@pytest.mark.smoke
def test_awq_with_kv_cache_quantization():
    """Test AWQModifier with kv_cache_scheme runs without errors"""
    model_id = "nm-testing/tinysmokellama-3.2"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
    dataset = Dataset.from_dict({"text": ["Hello world " * 10]})

    recipe = [
        AWQModifier(),
        QuantizationModifier(
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=8,
                        type=QuantizationType.INT,
                        strategy=QuantizationStrategy.GROUP,
                        group_size=32,
                        symmetric=True,
                        dynamic=False,
                    ),
                )
            },
            kv_cache_scheme=QuantizationArgs(
                num_bits=8,
                type=QuantizationType.FLOAT,
                strategy=QuantizationStrategy.TENSOR,
                symmetric=True,
                dynamic=False,
            ),
        ),
    ]

    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=16,
        num_calibration_samples=1,
    )

    # Verify quantization was applied
    assert hasattr(model.model.layers[0].self_attn, "kv_cache")


@requires_gpu(1)
@pytest.mark.smoke
def test_awq_raises_on_non_finite_identity_grid_loss():
    """Test AWQModifier raises when the identity grid loss is non-finite"""
    model_id = "nm-testing/tinysmokellama-3.2"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
    dataset = Dataset.from_dict({"text": ["Hello world " * 10]})

    self_attn = model.model.layers[0].self_attn
    forward_count = 0

    # hook will set weights to NaN after first forward pass,
    # so fp16 baseline has reasonable output but first scales
    # search does not, to test this specific edge case
    def corrupt_weights_after_baseline(module, _inputs, _output):
        nonlocal forward_count
        forward_count += 1
        if forward_count == 2:
            module.q_proj.weight.data.fill_(float("nan"))

    handle = self_attn.register_forward_hook(corrupt_weights_after_baseline)
    recipe = [
        AWQModifier(n_grid=3),
        QuantizationModifier(
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=8,
                        type=QuantizationType.INT,
                        strategy=QuantizationStrategy.GROUP,
                        group_size=32,
                        symmetric=True,
                        dynamic=False,
                    ),
                )
            },
        ),
    ]

    try:
        with pytest.raises(Exception, match="No finite loss"):
            oneshot(
                model=model,
                dataset=dataset,
                recipe=recipe,
                max_seq_length=16,
                num_calibration_samples=1,
            )
    finally:
        handle.remove()


@pytest.mark.unit
@pytest.mark.parametrize(
    "n_grid, duo_scaling",
    [
        (20, True),
        (10, False),
        (30, "both"),
    ],
)
def test_get_grid_search_params(n_grid, duo_scaling):
    awq = AWQModifier(n_grid=n_grid, duo_scaling=duo_scaling)

    grid_search_params = awq._get_grid_search_params()

    assert (
        0.0,
        False,
    ) in grid_search_params, "Identity scale case (0.0, False) should be in grid search"

    assert all(ratio >= 0.0 for ratio, _ in grid_search_params), "ratio<0 found"
    assert all(ratio <= 1.0 for ratio, _ in grid_search_params), "ratio>1 found"

    assert (
        len(grid_search_params) == n_grid
    ), f"Expected {n_grid} grid points, found {len(grid_search_params)}"

    n_false = len(
        [ratio for (ratio, duo_scaling) in grid_search_params if duo_scaling is False]
    )
    n_true = len(
        [ratio for (ratio, duo_scaling) in grid_search_params if duo_scaling is True]
    )
    if duo_scaling is False:
        assert n_false == n_grid
        assert n_true == 0
    elif duo_scaling is True:
        # duo_scaling True is hard-coded to include (0.0, False) case
        assert n_false == 1
        assert n_true == n_grid - 1
    else:
        assert abs(n_false - n_true) <= 1


@requires_compute_capability(9, 0)  # Requires H100 or higher
@torch.no_grad()
def test_awq_nvfp4_saves_fused_global_scale(tmp_path):
    """
    Test that AWQ with NVFP4 (TENSOR_GROUP) properly saves and fuses global_scale.

    This is a regression test for bugs where:
    1. get_qparams() was iterated without .items()
    2. qparams were assigned to wrong attributes (scale instead of weight_scale)

    Requires H100+ GPU for NVFP4 support.
    """
    from transformers import AutoModelForCausalLM

    from llmcompressor import oneshot

    model_id = "nm-testing/tinysmokellama-3.2"
    output = tmp_path / "nvfp4_awq_output"
    device = 0 if torch.accelerator.is_available() else "cpu"

    # NVFP4 AWQ recipe targeting one layer
    recipe = [
        AWQModifier(duo_scaling=False),
        QuantizationModifier(
            scheme="NVFP4",
            targets=["Linear"],
            ignore=["lm_head", "re:model\\.layers\\.(?!0\\.).*"],  # Only layer 0
        ),
    ]

    # Quantize
    oneshot(
        model=model_id,
        dataset="open_platypus",
        output_dir=output,
        recipe=recipe,
        num_calibration_samples=8,
        splits={"calibration": "train[:8]"},
    )

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(output, device_map=device)

    # Check layer 0 has global_scale attributes
    layer_0 = model.model.layers[0]

    # Check QKV
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        proj = getattr(layer_0.self_attn, proj_name)
        assert hasattr(
            proj, "weight_global_scale"
        ), f"{proj_name} missing weight_global_scale"

        gs = proj.weight_global_scale.item()
        assert gs > 0, f"{proj_name} global_scale should be positive, got {gs}"
        assert gs > 1e-10, f"{proj_name} global_scale too small: {gs}"
        assert gs < 1e10, f"{proj_name} global_scale too large: {gs}"

    # Verify QKV global_scales are fused (identical)
    q_gs = layer_0.self_attn.q_proj.weight_global_scale.item()
    k_gs = layer_0.self_attn.k_proj.weight_global_scale.item()
    v_gs = layer_0.self_attn.v_proj.weight_global_scale.item()

    assert abs(q_gs - k_gs) < 1e-6, f"QKV not fused: Q={q_gs}, K={k_gs}"
    assert abs(k_gs - v_gs) < 1e-6, f"QKV not fused: K={k_gs}, V={v_gs}"

    # Check gate/up
    for proj_name in ["gate_proj", "up_proj"]:
        proj = getattr(layer_0.mlp, proj_name)
        assert hasattr(
            proj, "weight_global_scale"
        ), f"{proj_name} missing weight_global_scale"

        gs = proj.weight_global_scale.item()
        assert gs > 0, f"{proj_name} global_scale should be positive, got {gs}"
        assert gs > 1e-10, f"{proj_name} global_scale too small: {gs}"
        assert gs < 1e10, f"{proj_name} global_scale too large: {gs}"

    # Verify gate/up global_scales are fused (identical)
    gate_gs = layer_0.mlp.gate_proj.weight_global_scale.item()
    up_gs = layer_0.mlp.up_proj.weight_global_scale.item()

    assert abs(gate_gs - up_gs) < 1e-6, f"gate/up not fused: gate={gate_gs}, up={up_gs}"

    # Verify QKV and gate/up are NOT fused together
    assert abs(q_gs - gate_gs) > 1e-6, f"QKV and gate/up incorrectly fused: {q_gs}"


class _TinyGQAAttention(torch.nn.Module):
    """
    Minimal grouped query attention block.

    The output of `v_proj` is repeated to the query dimension (`repeat_kv`) before
    the output projection, which is the layout a `v_proj -> o_proj` mapping has to
    account for.
    """

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, head_dim):
        super().__init__()
        self.num_repeats = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        self.v_proj = Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = Linear(num_attention_heads * head_dim, hidden_size, bias=False)

    def repeat_kv(self, hidden_states):
        num_kv_heads = hidden_states.size(-1) // self.head_dim
        expanded = hidden_states.reshape(
            *hidden_states.shape[:-1], num_kv_heads, 1, self.head_dim
        ).expand(
            *hidden_states.shape[:-1], num_kv_heads, self.num_repeats, self.head_dim
        )
        return expanded.reshape(
            *hidden_states.shape[:-1], num_kv_heads * self.num_repeats * self.head_dim
        )

    def forward(self, hidden_states):
        return self.o_proj(self.repeat_kv(self.v_proj(hidden_states)))


class _TinyGQAConfig:
    """Attention sizes of `_TinyGQAAttention`, in the shape of a hf config"""

    def __init__(self, num_attention_heads=4, num_key_value_heads=2, head_dim=2):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = 8


def _make_gqa_attention(
    with_config=True, num_attention_heads=4, num_key_value_heads=2, head_dim=2
):
    """
    4 query heads, 2 kv heads and head size 2, so that v_proj has 4 outputs and
    o_proj consumes 8 inputs. The block is wrapped so that its projections are named
    `self_attn.v_proj` and `self_attn.o_proj`, which is how the awq mappings look
    them up.
    """
    attention = _TinyGQAAttention(
        hidden_size=8,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    model = torch.nn.ModuleDict({"self_attn": attention})
    if with_config:
        model.config = _TinyGQAConfig(
            num_attention_heads, num_key_value_heads, head_dim
        )
    return model


# per channel group quantization is required for the smoothing factors to have an
# effect on the quantization error
_GQA_QUANT_SCHEME = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=4,
        symmetric=False,
        dynamic=False,
    ),
)


@pytest.mark.unit
def test_gqa_v_proj_o_proj_mapping_is_resolved():
    """
    A v_proj -> o_proj mapping of a grouped query attention model used to be
    skipped because the number of query heads is a multiple of the number of kv
    heads instead of being equal to it
    """
    model = _make_gqa_attention()
    awq = AWQModifier(
        mappings=[AWQMapping("re:.*v_proj", ["re:.*o_proj"])],
        grouped_query_smoothing=True,
    )
    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
    )

    awq._set_resolved_mappings(model)

    assert len(awq._resolved_mappings) == 1
    mapping = awq._resolved_mappings[0]
    assert mapping.smooth_name == "self_attn.v_proj"
    assert mapping.balance_names == ["self_attn.o_proj"]
    assert mapping.head_dim == 2


@pytest.mark.unit
def test_grouped_query_smoothing_is_disabled_by_default():
    """
    Smoothing value projections over the repeat groups of grouped query attention is
    opt in: the mapping is left alone unless the modifier asks for it, so that the
    behaviour of existing recipes does not change
    """
    model = _make_gqa_attention()
    awq = AWQModifier(mappings=[AWQMapping("re:.*v_proj", ["re:.*o_proj"])])
    assert awq.grouped_query_smoothing is False
    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
    )

    awq._set_resolved_mappings(model)

    assert len(awq._resolved_mappings) == 0


@pytest.mark.unit
def test_gqa_v_proj_o_proj_mapping_requires_head_dim():
    """
    Without a config stating the attention head size the repeat groups of the kv
    heads cannot be reconstructed, so the mapping has to be skipped as before
    rather than smoothed incorrectly
    """
    model = _make_gqa_attention(with_config=False)
    awq = AWQModifier(
        mappings=[AWQMapping("re:.*v_proj", ["re:.*o_proj"])],
        grouped_query_smoothing=True,
    )
    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
    )

    awq._set_resolved_mappings(model)

    assert len(awq._resolved_mappings) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_attention_heads, num_key_value_heads, head_dim, config_head_dim, expected",
    [
        (4, 3, 2, 2, False),  # query dim is not a whole number of kv repeats
        (4, 2, 3, 2, False),  # head size does not describe the kv projection
        (4, 2, 2, 3, False),  # head size does not divide either dimension
        (4, 2, 2, 2, True),  # 2 kv heads of size 2, repeated twice
        (4, 4, 2, 2, True),  # no repeats, shapes match and are handled as before
    ],
)
def test_gqa_v_proj_o_proj_mapping_shape_alignment(
    num_attention_heads, num_key_value_heads, head_dim, config_head_dim, expected
):
    """
    Shapes which could be mistaken for grouped query attention, but where the head
    size does not align the kv and query dimensions, must still be skipped
    """
    model = _make_gqa_attention(
        with_config=False,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    model.config = _TinyGQAConfig(
        num_attention_heads, num_key_value_heads, config_head_dim
    )
    awq = AWQModifier(
        mappings=[AWQMapping("re:.*v_proj", ["re:.*o_proj"])],
        grouped_query_smoothing=True,
    )
    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
    )

    awq._set_resolved_mappings(model)

    assert len(awq._resolved_mappings) == int(expected)


@pytest.mark.unit
def test_gqa_mapping_requires_output_projections():
    """
    The gqa handling assumes every balance layer consumes the repeated output of the
    smooth layer. A mapping which also balances a layer of kv width, which cannot
    carry the expanded scales, has to be left alone
    """
    attention = torch.nn.ModuleDict(
        {
            "v_proj": Linear(8, 4, bias=False),
            "o_proj": Linear(8, 8, bias=False),
            "z_proj": Linear(4, 8, bias=False),
        }
    )
    model = torch.nn.ModuleDict({"self_attn": attention})
    model.config = _TinyGQAConfig()
    awq = AWQModifier(
        mappings=[AWQMapping("re:.*v_proj", ["re:.*o_proj", "re:.*z_proj"])],
        grouped_query_smoothing=True,
    )
    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
    )

    awq._set_resolved_mappings(model)

    assert len(awq._resolved_mappings) == 0


@pytest.mark.unit
def test_fused_qkv_o_proj_mapping_is_rejected_for_gqa():
    """
    Fused qkv_proj is not supported for grouped query attention, its v_proj block
    occupies a third of the fused output features which does not match the repeat
    groups of an arbitrary model
    """

    class _TinyFusedAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = Linear(8, 16, bias=False)
            self.o_proj = Linear(8, 8, bias=False)

    model = torch.nn.ModuleDict({"self_attn": _TinyFusedAttention()})
    model.config = _TinyGQAConfig()
    awq = AWQModifier(
        mappings=[AWQMapping("re:.*qkv_proj", ["re:.*o_proj"])],
        grouped_query_smoothing=True,
    )
    apply_quantization_config(
        model,
        config=QuantizationModifier(scheme="W4A16_ASYM").resolve_quantization_config(),
    )

    awq._set_resolved_mappings(model)

    assert len(awq._resolved_mappings) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "config, kv_dim, query_dim, expected",
    [
        (_TinyGQAConfig(), 4, 8, 2),
        ({"num_key_value_heads": 2, "num_attention_heads": 4}, 4, 8, 2),
        ({"head_dim": 2}, 4, 8, 2),
        ({"head_dim": 0}, 4, 8, None),
        ({"head_dim": -2}, 4, 8, None),
        (_TinyGQAConfig(head_dim=3), 4, 8, None),  # 4 != 2 kv heads * head size 3
        (_TinyGQAConfig(), 4, 6, None),  # 6 != 4 query heads * head size 2
        (None, 4, 8, None),
        # head size missing, recovered from the kv projection width
        (
            type("Config", (), {"num_key_value_heads": 2, "num_attention_heads": 4}),
            4,
            8,
            2,
        ),
        (
            type("Config", (), {"num_key_value_heads": 2, "num_attention_heads": 4}),
            4,
            6,
            None,
        ),
        # neither head size nor kv heads can be determined
        (type("Config", (), {"hidden_size": 8, "num_attention_heads": 4}), 4, 8, None),
    ],
)
def test_get_head_dim(config, kv_dim, query_dim, expected):
    model = torch.nn.Module()
    if config is not None:
        model.config = config
    assert _get_head_dim(model, kv_dim, query_dim) == expected


@pytest.mark.unit
@pytest.mark.parametrize("outer_dict", [False, True])
@pytest.mark.parametrize("inner_dict", [False, True])
def test_get_head_dim_from_nested_text_config(outer_dict, inner_dict):
    """Multimodal configs nest the attention sizes under text_config"""
    model = torch.nn.Module()
    text_config = vars(_TinyGQAConfig()) if inner_dict else _TinyGQAConfig()
    model.config = (
        {"text_config": text_config}
        if outer_dict
        else type("Config", (), {"text_config": text_config})()
    )
    assert _get_head_dim(model, 4) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "kv_dim, hidden_dim, head_dim, expected",
    [
        (4, 8, 2, True),  # 2 repeats of 2 wide kv heads
        (4, 8, None, False),  # head size unknown
        (8, 8, 2, False),  # no repeats, handled by the equality check
        (4, 6, 2, False),  # not a whole number of repeats
        (6, 8, 3, False),  # head size does not divide the dimensions evenly
        (4, 8, 3, False),
        (None, 8, 2, False),  # smooth layer is not a projection
        (4, None, 2, False),  # balance layer is not a projection
        (0, 8, 2, False),
        (-4, 8, 2, False),
        (4, 8, 0, False),
        (4, 8, -2, False),
        (4, 0, 2, False),
        (4, -8, 2, False),
    ],
)
def test_is_gqa_projection(kv_dim, hidden_dim, head_dim, expected):
    assert _is_gqa_projection(kv_dim, hidden_dim, head_dim) == expected


@pytest.mark.unit
def test_compress_scales_for_gqa():
    """
    Statistics of the query dimension are averaged over the kv heads they are
    repeated from, per attention head
    """
    # 2 kv heads, 2 repeats, head size 3 -> kv_dim = 6, hidden_dim = 12
    # layout after repeat_kv: [kv0_r0(3), kv0_r1(3), kv1_r0(3), kv1_r1(3)]
    scales = torch.arange(12, dtype=torch.float)
    compressed = _compress_scales_for_gqa(scales, kv_dim=6, head_dim=3)

    assert compressed.shape == (6,)
    # kv0 = mean([0, 1, 2], [3, 4, 5]), kv1 = mean([6, 7, 8], [9, 10, 11])
    assert_close(compressed, torch.tensor([1.5, 2.5, 3.5, 7.5, 8.5, 9.5]))


@pytest.mark.unit
def test_expand_scales_for_gqa():
    """
    Scales of the kv dimension are repeated over the query heads reading from them
    """
    scales = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expanded = _expand_scales_for_gqa(scales, query_dim=12, head_dim=3)

    assert expanded.shape == (12,)
    expected = torch.tensor(
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
    )
    assert_close(expanded, expected)


@pytest.mark.unit
def test_compress_expand_scales_roundtrip():
    kv_scales = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expanded = _expand_scales_for_gqa(kv_scales, query_dim=12, head_dim=3)
    compressed = _compress_scales_for_gqa(expanded, kv_dim=6, head_dim=3)
    assert_close(compressed, kv_scales)


@pytest.mark.unit
def test_gqa_scale_grouping_matches_repeat_kv():
    """
    The gqa handling assumes the layout that models produce with `repeat_kv`, which
    repeats every kv head contiguously. This pins that assumption against the
    implementation the models actually use, so that a model laying its query heads out
    differently cannot silently be grouped the wrong way
    """
    from transformers.models.llama.modeling_llama import repeat_kv

    num_heads, num_kv_heads, head_dim = 4, 2, 3
    num_repeats = num_heads // num_kv_heads
    kv_dim = num_kv_heads * head_dim

    # feed the kv channel index as a value: `repeat_kv` stamps the source channel of
    # every query channel into the result
    kv_channels = torch.arange(kv_dim, dtype=torch.float)
    library_layout = repeat_kv(
        kv_channels.reshape(1, num_kv_heads, 1, head_dim), num_repeats
    ).reshape(-1)

    assert_close(
        _expand_scales_for_gqa(
            kv_channels, query_dim=num_heads * head_dim, head_dim=head_dim
        ),
        library_layout,
    )


def _setup_gqa_smoothing():
    """
    Build a grouped query attention block with awq applied to its v_proj -> o_proj
    mapping, ready for `_apply_smoothing`
    """
    model = _make_gqa_attention()
    awq = AWQModifier(
        mappings=[AWQMapping("re:.*v_proj", ["re:.*o_proj"])],
        grouped_query_smoothing=True,
    )
    awq.offload_device = None
    apply_quantization_config(
        model,
        config=QuantizationModifier(
            config_groups={"group_0": _GQA_QUANT_SCHEME}
        ).resolve_quantization_config(),
    )
    awq._set_resolved_mappings(model)
    assert len(awq._resolved_mappings) == 1, "v_proj -> o_proj should not be skipped"
    return model, awq, awq._resolved_mappings[0]


def _record_calibration(awq, mapping, hidden_states, model):
    """
    Populate the caches that the calibration hooks would have populated: the parent
    kwargs and the absolute input activations of the balance layer
    """
    attention = model["self_attn"]
    o_proj_input = attention.repeat_kv(attention.v_proj(hidden_states))
    awq._parent_args_cache[mapping.parent] = IntermediatesCache(
        None, awq.offload_device
    )
    awq._parent_args_cache[mapping.parent].append({"input": o_proj_input})
    flat_activations = o_proj_input.abs().flatten(0, -2)
    awq._smooth_activation_stats[mapping.smooth_name] = [
        flat_activations.sum(0),
        torch.tensor(flat_activations.size(0)),
    ]


def _block_output(attention, hidden_states):
    """
    Unquantized output of the block. The quantization hooks are bypassed on purpose,
    the smoothing is only expected to leave the fp16 function of the block untouched
    """
    values = torch.nn.functional.linear(hidden_states, attention.v_proj.weight)
    return torch.nn.functional.linear(
        attention.repeat_kv(values), attention.o_proj.weight
    )


def _assert_output_unchanged(output, expected):
    """
    The block output has to survive the smoothing up to floating point error. The
    tolerance is relative to the output magnitude because the smoothing factors
    spread the intermediate products, which amplifies fp32 rounding. Scales which
    are not constant within a repeat group change the output by orders of magnitude
    more than this, so the check still catches them
    """
    assert_close(output, expected, atol=1e-4 * expected.abs().max(), rtol=1e-3)


@pytest.mark.unit
@torch.no_grad()
def test_compute_best_scale_of_non_projection_smooth_layer():
    """
    The default mappings include input_layernorm -> q/k/v_proj, whose smooth layer is
    not a projection and therefore has no kv dimension. Such a mapping must not be
    treated as grouped query attention, even though the model config does state a
    head size
    """
    attention = torch.nn.ModuleDict(
        {
            "input_layernorm": torch.nn.LayerNorm(8),
            "q_proj": Linear(8, 8, bias=False),
        }
    )
    model = torch.nn.ModuleDict({"self_attn": attention})
    model.config = _TinyGQAConfig()
    awq = AWQModifier(
        mappings=[AWQMapping("re:.*input_layernorm", ["re:.*q_proj"])],
        duo_scaling=False,
    )
    awq.offload_device = None
    apply_quantization_config(
        model,
        config=QuantizationModifier(
            config_groups={"group_0": _GQA_QUANT_SCHEME}
        ).resolve_quantization_config(),
    )
    awq._set_resolved_mappings(model)
    assert len(awq._resolved_mappings) == 1
    mapping = awq._resolved_mappings[0]

    hidden_states = torch.randn(2, 4, 8)
    q_proj_input = attention["input_layernorm"](hidden_states)
    fp16_outputs = [attention["q_proj"](input=q_proj_input)]
    awq._parent_args_cache[mapping.parent] = IntermediatesCache(
        None, awq.offload_device
    )
    awq._parent_args_cache[mapping.parent].append({"input": q_proj_input})
    flat_activations = q_proj_input.abs().flatten(0, -2)
    awq._smooth_activation_stats[mapping.smooth_name] = [
        flat_activations.sum(0),
        torch.tensor(flat_activations.size(0)),
    ]

    scales = awq._compute_best_scale(
        mapping,
        fp16_outputs,
        {mapping.balance_layers[0]: mapping.balance_layers[0].weight.clone()},
    )

    assert scales.size(0) == 8


@pytest.mark.unit
@torch.no_grad()
def test_gqa_smoothing_preserves_block_output(monkeypatch):
    """
    Smoothing a grouped query attention block has to leave the unquantized output
    untouched: `o_proj` input channels are multiplied by the scales while `v_proj`
    output channels are divided by them, so the scales of the query channels have
    to be constant within every repeat group of a kv head.

    This runs the whole smoothing path (`_compute_best_scale` and the weight
    updates) rather than only the mapping resolution.
    """
    torch.manual_seed(0)
    model, awq, mapping = _setup_gqa_smoothing()
    attention = model["self_attn"]

    hidden_states = torch.randn(4, 32, 8)
    _record_calibration(awq, mapping, hidden_states, model)
    expected_output = _block_output(attention, hidden_states)

    # the grid search has to run in the kv dimension, scales of the query dimension
    # are not representable by v_proj
    reductions = []
    compress = awq_base._compress_scales_for_gqa

    def record_reduction(scales, target_dim, head_dim):
        reductions.append((scales.size(0), target_dim, head_dim))
        return compress(scales, target_dim, head_dim)

    monkeypatch.setattr(awq_base, "_compress_scales_for_gqa", record_reduction)

    orig_v_proj = attention.v_proj.weight.clone()
    orig_o_proj = attention.o_proj.weight.clone()
    awq._apply_smoothing(model)

    assert reductions, "expected the statistics to be reduced to the kv dimension"
    assert all(
        reduction == (8, 4, 2) for reduction in reductions
    ), f"expected query dim 8 to be reduced to kv dim 4, got {reductions}"
    # duo scaling is on by default, so the weight means are reduced as well
    assert len(reductions) == 2

    output = _block_output(attention, hidden_states)
    _assert_output_unchanged(output, expected_output)

    # v_proj divides its output channels, o_proj multiplies its input channels
    scales = orig_v_proj[:, 0] / attention.v_proj.weight[:, 0]
    applied = attention.o_proj.weight[0] / orig_o_proj[0]
    expected_applied = scales.reshape(2, 2).repeat_interleave(2, dim=0).reshape(-1)
    assert_close(applied, expected_applied, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "smooth_norm,smooth_value", list(product([False, True], repeat=2))
)
@torch.no_grad()
def test_gqa_shared_value_mapping_coordinates(monkeypatch, smooth_norm, smooth_value):
    """Default-order transforms preserve the cached O input and the FP function.

    Fixed nonuniform scales isolate coordinate consistency from the quantization
    search objective. This does not assert that either transform improves accuracy.
    """

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = Linear(8, 8, bias=False)
            self.k_proj = Linear(8, 4, bias=False)
            self.v_proj = Linear(8, 4, bias=True)
            self.o_proj = Linear(8, 8, bias=False)

        def forward(self, hidden_states):
            from transformers.models.llama.modeling_llama import repeat_kv

            batch, length, _ = hidden_states.shape
            query = self.q_proj(hidden_states).view(batch, length, 4, 2)
            key = self.k_proj(hidden_states).view(batch, length, 2, 2)
            value = self.v_proj(hidden_states).view(batch, length, 2, 2)
            query = query.transpose(1, 2)
            key = repeat_kv(key.transpose(1, 2), 2)
            value = repeat_kv(value.transpose(1, 2), 2)
            probs = (query @ key.transpose(-1, -2) / 2**0.5).softmax(-1)
            context = (probs @ value).transpose(1, 2).reshape(batch, length, 8)
            return self.o_proj(context)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _TinyGQAConfig()
            self.input_layernorm = torch.nn.LayerNorm(8)
            self.self_attn = Attention()

        def forward(self, hidden_states):
            return self.self_attn(self.input_layernorm(hidden_states))

    torch.manual_seed(104)
    model = Block()
    apply_quantization_config(
        model,
        config=QuantizationModifier(
            config_groups={"group_0": _GQA_QUANT_SCHEME}
        ).resolve_quantization_config(),
    )
    for module in model.modules():
        module.quantization_enabled = False
    mappings = []
    if smooth_norm:
        mappings.append(
            AWQMapping(
                "input_layernorm",
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            )
        )
    if smooth_value:
        mappings.append(AWQMapping("self_attn.v_proj", ["self_attn.o_proj"]))
    awq = AWQModifier(mappings=mappings, grouped_query_smoothing=True)
    awq.offload_device = None
    awq._set_resolved_mappings(model)
    assert [m.smooth_name for m in awq._resolved_mappings] == [
        m.smooth_layer for m in mappings
    ]
    awq._setup_activation_cache_hooks()
    inputs = torch.randn(2, 7, 8)
    expected = model(inputs)
    awq.remove_hooks()

    norm_scales = torch.tensor([0.25, 2, 0.5, 4, 1, 3, 0.75, 1.5])
    value_scales = torch.tensor([0.5, 2, 4, 0.25])
    original_value = model.self_attn.v_proj.weight.clone()
    original_bias = model.self_attn.v_proj.bias.clone()
    searched = []

    def fixed_scale(self, mapping, fp16_outputs, orig_layer_weights):
        searched.append(mapping.smooth_name)
        if mapping.smooth_name == "input_layernorm":
            assert mapping.parent is model.self_attn
            return norm_scales

        assert mapping.parent is model.self_attn.o_proj
        # The preceding norm -> QKV transform is function preserving, including
        # the V output; O's cached input must still match a fresh forward pass.
        current_inputs = []
        handle = model.self_attn.o_proj.register_forward_pre_hook(
            lambda module, args: current_inputs.append(args[0].clone())
        )
        try:
            assert_close(model(inputs), expected)
        finally:
            handle.remove()
        cached = next(self._parent_args_cache[mapping.parent].iter_prefetch())
        assert_close(cached["input"], current_inputs[0])
        assert_close(fp16_outputs[0], expected)
        return value_scales

    monkeypatch.setattr(AWQModifier, "_compute_best_scale", fixed_scale)
    awq._apply_smoothing(model)

    assert searched == [m.smooth_layer for m in mappings]
    assert_close(model(inputs), expected)
    expected_value = original_value
    if smooth_norm:
        expected_value = expected_value * norm_scales.view(1, -1)
    if smooth_value:
        expected_value = expected_value / value_scales.view(-1, 1)
    assert_close(model.self_attn.v_proj.weight, expected_value)
    assert_close(
        model.self_attn.v_proj.bias,
        original_bias / value_scales if smooth_value else original_bias,
    )
    assert not awq._parent_args_cache
    assert not awq._smooth_activation_stats


@pytest.mark.unit
@torch.no_grad()
def test_gqa_smoothing_applies_group_constant_scales(monkeypatch):
    """
    Pin the weight updates to a known set of scales: the kv channels of `v_proj` are
    divided by the scales, while the query channels of `o_proj` are multiplied by
    them repeated over the kv heads they were expanded from
    """
    torch.manual_seed(0)
    model, awq, mapping = _setup_gqa_smoothing()
    attention = model["self_attn"]

    kv_scales = torch.tensor([0.5, 2.0, 4.0, 0.25])
    monkeypatch.setattr(
        AWQModifier, "_compute_best_scale", lambda *args, **kwargs: kv_scales
    )

    hidden_states = torch.randn(4, 32, 8)
    _record_calibration(awq, mapping, hidden_states, model)
    expected_output = _block_output(attention, hidden_states)
    orig_v_proj = attention.v_proj.weight.clone()
    orig_o_proj = attention.o_proj.weight.clone()

    awq._apply_smoothing(model)

    assert_close(attention.v_proj.weight, orig_v_proj / kv_scales.view(-1, 1))
    # kv0 = [0.5, 2.0] and kv1 = [4.0, 0.25] are both repeated twice, in the same
    # order as `repeat_kv` lays out the query heads
    expanded = torch.tensor([0.5, 2.0, 0.5, 2.0, 4.0, 0.25, 4.0, 0.25])
    assert_close(attention.o_proj.weight, orig_o_proj * expanded.view(1, -1))

    output = _block_output(attention, hidden_states)
    _assert_output_unchanged(output, expected_output)
