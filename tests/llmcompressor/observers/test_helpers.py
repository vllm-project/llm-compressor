import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.observers.helpers import flatten_for_calibration


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
    "args",
    [
        QuantizationArgs(strategy="tensor"),
        QuantizationArgs(strategy="channel"),
        QuantizationArgs(strategy="group", group_size=4),
        QuantizationArgs(strategy="tensor_group", group_size=4),
        QuantizationArgs(strategy="block", block_structure=[5, 4]),
        # When block structure does not evenly divide module.weight
        QuantizationArgs(strategy="block", block_structure=[3, 3]),
    ],
)
def test_flatten_for_calibration_weights(args):
    module = torch.nn.Linear(8, 10)
    scheme = QuantizationScheme(targets=[], weights=args)
    initialize_module_for_quantization(module, scheme)

    weight_flattened = flatten_for_calibration(
        module.weight,
        "weight",
        scheme.weights,
    )
    assert weight_flattened.shape[1:-1] == module.weight_scale.shape
    assert weight_flattened.shape[1:-1] == module.weight_zero_point.shape
