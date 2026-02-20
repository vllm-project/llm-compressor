import pytest

import torch

from llmcompressor.modifiers.experimental import TensorizedLinear, BlockTensorizedLinear


@pytest.mark.parametrize(
    "in_features,out_features,block_size,num_cores,bias",
    [
        (32, 64, None, 2, True),
        (32, 64, None, 2, False),
        (64, 32, 16, 2, False),
        (128, 64, 16, 3, True),
    ],
)
def test_tensorized_linear_reconstructs_weight_and_output(
    in_features, out_features, block_size, num_cores, bias
):
    batch_size = 7
    in_features = 32
    out_features = 64

    linear = torch.nn.Linear(in_features, out_features, bias=bias)

    tensorized_linear = (
        TensorizedLinear.from_linear(linear, num_cores=num_cores, rank="same")
        if block_size is None
        else BlockTensorizedLinear.from_linear(
            linear, block_size, num_cores=num_cores, rank="same"
        )
    )

    orig_weight = linear.weight.data
    tensorized_weight = tensorized_linear.to_matrix()
    assert (
        orig_weight - tensorized_weight
    ).abs().max() < 1e-6, "Weights not reconstructing"

    inpt = torch.rand((batch_size, in_features), requires_grad=False)
    orig_output = linear(inpt)
    tensorized_output = tensorized_linear(inpt)

    assert (
        orig_output.shape == tensorized_output.shape
    ), "Output activations have incorrect shape"
    (orig_output - tensorized_output).abs().max() < 1e-6, "Outputs not reconstructing"
