from unittest.mock import MagicMock

import torch

from llmcompressor.modeling.granite4 import GraniteMoeHybridParallelExpertsLinear
from tests.testing_utils import requires_transformers_v4

pytestmark = requires_transformers_v4


def _make_layer(
    num_experts, output_size, input_size, weight_shape, scale_shape, zp_shape=None
):
    """Create a mock layer with the given shapes to test to_3d_expert."""
    layer = MagicMock(spec=GraniteMoeHybridParallelExpertsLinear)
    layer.num_experts = num_experts
    layer.output_size = output_size
    layer.input_size = input_size
    layer.weight = torch.nn.Parameter(torch.randn(weight_shape), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.randn(scale_shape), requires_grad=False
    )
    layer.is_2d = True
    if zp_shape is not None:
        layer.weight_zero_point = torch.nn.Parameter(
            torch.randn(zp_shape), requires_grad=False
        )
    else:
        # hasattr should return False for weight_zero_point
        del layer.weight_zero_point
    return layer


def test_to_3d_expert_int4_symmetric():
    """W4A16 symmetric: packed weight, per-channel scale, no zero_point."""
    num_experts, output_size, input_size = 4, 64, 128
    pack_factor = 8  # 4-bit packing
    layer = _make_layer(
        num_experts,
        output_size,
        input_size,
        weight_shape=(num_experts * output_size, input_size // pack_factor),
        scale_shape=(num_experts * output_size, 1),
    )
    GraniteMoeHybridParallelExpertsLinear.to_3d_expert(layer)
    assert layer.weight.shape == (
        num_experts,
        output_size,
        input_size // pack_factor,
    )
    assert layer.weight_scale.shape == (num_experts, output_size, 1)


def test_to_3d_expert_int4_asymmetric():
    """W4A16 asymmetric: packed weight + packed zero_point on dim0."""
    num_experts, output_size, input_size = 4, 64, 128
    pack_factor = 8
    layer = _make_layer(
        num_experts,
        output_size,
        input_size,
        weight_shape=(num_experts * output_size, input_size // pack_factor),
        scale_shape=(num_experts * output_size, 1),
        zp_shape=(num_experts * output_size // pack_factor, 1),
    )
    GraniteMoeHybridParallelExpertsLinear.to_3d_expert(layer)
    assert layer.weight.shape == (
        num_experts,
        output_size,
        input_size // pack_factor,
    )
    assert layer.weight_scale.shape == (num_experts, output_size, 1)
    assert layer.weight_zero_point.shape == (
        num_experts,
        output_size // pack_factor,
        1,
    )


def test_to_3d_expert_fp8_block():
    """FP8 block quantization: grouped scale, no packing."""
    num_experts, output_size, input_size = 4, 64, 128
    group_size = 32
    num_row_groups = output_size  # per-row
    num_col_groups = input_size // group_size
    layer = _make_layer(
        num_experts,
        output_size,
        input_size,
        weight_shape=(num_experts * output_size, input_size),
        scale_shape=(num_experts * num_row_groups, num_col_groups),
    )
    GraniteMoeHybridParallelExpertsLinear.to_3d_expert(layer)
    assert layer.weight.shape == (num_experts, output_size, input_size)
    assert layer.weight_scale.shape == (
        num_experts,
        num_row_groups,
        num_col_groups,
    )


def test_to_3d_expert_fp8_per_channel():
    """FP8 per-channel: no packing, scale per row."""
    num_experts, output_size, input_size = 4, 64, 128
    layer = _make_layer(
        num_experts,
        output_size,
        input_size,
        weight_shape=(num_experts * output_size, input_size),
        scale_shape=(num_experts * output_size, 1),
    )
    GraniteMoeHybridParallelExpertsLinear.to_3d_expert(layer)
    assert layer.weight.shape == (num_experts, output_size, input_size)
    assert layer.weight_scale.shape == (num_experts, output_size, 1)
