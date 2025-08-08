import pytest
import torch

from llmcompressor.modeling.fuse import center_embeddings, fuse_norm_linears


@pytest.mark.unit
def test_center_embeddings():
    embedding = torch.nn.Embedding(10, 10)
    center_embeddings(embedding)

    assert torch.allclose(
        embedding.weight.mean(dim=1), torch.zeros(embedding.num_embeddings), atol=1e-5
    )


@pytest.mark.unit
def test_fuse_norm_linears():
    norm = torch.nn.LayerNorm((5,))
    norm.weight.data = torch.rand(norm.weight.shape)
    linears = [
        torch.nn.Linear(5, 5),
        torch.nn.Linear(5, 5),
    ]

    input = torch.rand((1, 5), requires_grad=False)
    true_output = torch.stack([linear(norm(input)) for linear in linears])

    fuse_norm_linears(norm, linears)
    output = torch.stack([linear(norm(input)) for linear in linears])

    assert torch.allclose(true_output, output)
