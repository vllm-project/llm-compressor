import pytest

import torch
import torch.nn.functional as F

from llmcompressor.modifiers.experimental import TensorizedLinear, BlockTensorizedLinear


@pytest.mark.parametrize(
    "batch_size,seq_len,in_features,out_features,block_size,num_cores,bias",
    [
        (5, None, 32, 64, None, 2, True),
        (6, 16, 32, 64, None, 2, False),
        (7, 32, 64, 32, 16, 4, False),
        (8, None, 128, 64, 16, 3, True),
        (5, None, 1024, 1024, 512, 3, True),
        (6, 8, 1024, 1024, 16, 3, False),
        (7, 16, 1024, 1024, None, 4, True),
        (8, 8, 1024, 1024, 512, 3, False),
    ],
)
def test_tensorized_linear_reconstructs_weight_and_output(
    batch_size, seq_len, in_features, out_features, block_size, num_cores, bias
):

    linear = torch.nn.Linear(in_features, out_features, bias=bias)

    tensorized_linear = (
        TensorizedLinear.from_linear(linear, num_cores=num_cores, rank=1.0)
        if block_size is None
        else BlockTensorizedLinear.from_linear(
            linear, block_size, num_cores=num_cores, rank=1.0
        )
    )

    orig_weight = linear.weight.data
    tensorized_weight = tensorized_linear.to_matrix()
    assert (
        similarity := F.cosine_similarity(orig_weight, tensorized_weight).mean()
    ) > 0.5, f"Reconstructed weights not similar: {similarity}"

    inpt = (
        torch.rand((batch_size, seq_len, in_features), requires_grad=False)
        if seq_len is not None
        else torch.rand((batch_size, in_features), requires_grad=False)
    )
    orig_output = linear(inpt)
    tensorized_output = tensorized_linear(inpt)

    assert (
        orig_output.shape == tensorized_output.shape
    ), "Output activations have incorrect shape"
    assert (
        similarity := F.cosine_similarity(orig_weight, tensorized_weight).mean()
    ) > 0.5, f"Reconstructed outputs not similar: {similarity}"

    dense_tensorized_output = tensorized_linear.dense_forward(inpt)
    assert (
        loss := F.mse_loss(dense_tensorized_output, tensorized_output)
    ) < 1e-6, f"Outputs of forward and dense_forward are incompatible: {loss}"
