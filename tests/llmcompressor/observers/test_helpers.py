import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.observers.helpers import flatten_for_calibration


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
