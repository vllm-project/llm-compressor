import pytest

import torch
import torch.nn.functional as F

from llmcompressor.modifiers.experimental import TensorizedLinear, BlockTensorizedLinear


@pytest.mark.parametrize(
    "batch_size,in_features,out_features,block_size,num_cores,bias",
    [
        (5, 32, 64, None, 2, True),
        (6, 32, 64, None, 2, False),
        (7, 64, 32, 16, 2, False),
        (8, 128, 64, 16, 3, True),
        (5, 1024, 1024, 16, 3, True),
        (6, 1024, 1024, 16, 3, False),
        (7, 1024, 1024, None, 4, True),
        (8, 2048, 2048, None, 3, False),
    ],
)
def test_tensorized_linear_reconstructs_weight_and_output(
    batch_size, in_features, out_features, block_size, num_cores, bias
):
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
        F.mse_loss(orig_weight, tensorized_weight) < 1e-4
    ), "Weights not reconstructing"

    inpt = torch.rand((batch_size, in_features), requires_grad=False)
    orig_output = linear(inpt)
    tensorized_output = tensorized_linear(inpt)

    assert (
        orig_output.shape == tensorized_output.shape
    ), "Output activations have incorrect shape"
    assert (
        F.mse_loss(orig_output, tensorized_output) < 1e-4
    ), "Outputs not reconstructing"

    dense_tensorized_output = tensorized_linear.dense_forward(inpt)
    assert (
        F.mse_loss(dense_tensorized_output, tensorized_output) < 1e-4
    ), "Outputs of forward and dense_forward are incompatible"
