import math
from itertools import product

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.observers.helpers import (
    _pad_to_block_size_with_mean,
    flatten_for_calibration,
)


def make_dummy_g_idx(columns: int, group_size: int) -> torch.Tensor:
    perm = torch.randperm(columns)
    return torch.tensor([index // group_size for index in range(columns)])[perm]


@pytest.mark.parametrize(
    "args",
    [
        QuantizationArgs(strategy="tensor"),
        QuantizationArgs(strategy="tensor_group", group_size=4),
    ],
)
def test_flatten_for_calibration_input(args):
    module = torch.nn.Linear(8, 10)
    scheme = QuantizationScheme(targets=[], input_activations=args)
    initialize_module_for_quantization(module, scheme)

    input = torch.empty((3, 5, 8))
    input_flattened = flatten_for_calibration(input, "input", scheme.input_activations)
    assert input_flattened.shape[1:-1] == module.input_scale.shape
    assert input_flattened.shape[1:-1] == module.input_zero_point.shape


@pytest.mark.parametrize(
    "args,g_idx",
    [
        (QuantizationArgs(strategy="tensor"), None),
        (QuantizationArgs(strategy="channel"), None),
        (QuantizationArgs(strategy="group", group_size=4), None),
        (QuantizationArgs(strategy="group", group_size=4), make_dummy_g_idx(8, 4)),
        (QuantizationArgs(strategy="tensor_group", group_size=4), None),
        (QuantizationArgs(strategy="block", block_structure=[5, 4]), None),
    ],
)
def test_flatten_for_calibration_weights(args, g_idx):
    module = torch.nn.Linear(8, 10)
    scheme = QuantizationScheme(targets=[], weights=args)
    initialize_module_for_quantization(module, scheme)

    weight_flattened = flatten_for_calibration(
        module.weight,
        "weight",
        scheme.weights,
        g_idx=g_idx,
    )
    assert weight_flattened.shape[1:-1] == module.weight_scale.shape
    assert weight_flattened.shape[1:-1] == module.weight_zero_point.shape


@pytest.mark.parametrize(
    "rows,cols,block_height,block_width",
    [
        (256, 256, 64, 128),  # Both dimensions divisible
        (100, 200, 64, 128),  # Both dimensions not divisible
        (256, 300, 128, 64),  # Only cols not divisible
        (300, 256, 64, 128),  # Only rows not divisible
        (57, 47, 128, 64),  # Both dimensions smaller than block size
    ],
)
def test_pad_to_block_size_with_mean(rows, cols, block_height, block_width):
    value = torch.rand(rows, cols)

    block_rows = math.ceil(rows / block_height)
    block_cols = math.ceil(cols / block_width)

    new_value = _pad_to_block_size_with_mean(value, (block_height, block_width))
    new_rows, new_cols = new_value.shape

    assert (
        new_rows % block_height == 0
    ), f"new tensor has incompatible num_rows {new_rows} ({block_height})"
    assert (
        new_cols % block_width == 0
    ), f"new tensor has incompatible num_cols {new_cols} ({block_width})"

    assert torch.equal(
        value, new_value[: value.shape[0], : value.shape[1]]
    ), "new tensor doesn't have same values as original where appropriate"

    for block_row_idx, block_col_idx in product(range(block_rows), range(block_cols)):
        value_block = value[
            block_row_idx * block_height : min(
                value.shape[0], (1 + block_row_idx) * block_height
            ),
            block_col_idx * block_width : min(
                value.shape[1], (1 + block_col_idx) * block_width
            ),
        ]
        new_value_block = new_value[
            block_row_idx * block_height : (1 + block_row_idx) * block_height,
            block_col_idx * block_width : (1 + block_col_idx) * block_width,
        ]

        assert torch.equal(
            value_block.min(), new_value_block.min()
        ), f"Block {block_row_idx}x{block_col_idx} min vals have been altered"
        assert torch.equal(
            value_block.max(), new_value_block.max()
        ), f"Block {block_row_idx}x{block_col_idx} max vals have been altered"

        assert torch.isclose(
            value_block.mean(), new_value_block.mean()
        ), f"Block {block_row_idx}x{block_col_idx} mean vals have been altered"
