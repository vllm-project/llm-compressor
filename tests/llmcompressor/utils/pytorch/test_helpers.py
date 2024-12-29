import pytest
import torch
from torch import Tensor

from llmcompressor.utils.pytorch.helpers import mse_loss_with_chunking, reclaim_memory
from tests.testing_utils import requires_gpu


# Test the mse_loss_with_chunking function
@pytest.fixture
def tensors():
    tensor_a = torch.randn(3, 5, requires_grad=True)
    tensor_b = torch.randn(3, 5)
    return tensor_a, tensor_b


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_mse_loss_with_chunking_correctness(
    tensors: tuple[Tensor, Tensor], device: torch.device
):
    tensor_a, tensor_b = tensors
    loss = mse_loss_with_chunking(tensor_a, tensor_b, device)
    expected_loss = (
        (tensor_a - tensor_b).float().pow(2).sum() / tensor_a.numel()
    ).item()
    assert pytest.approx(loss) == expected_loss


def test_mse_loss_with_chunking_with_chunk_memory_correctness(
    tensors: tuple[Tensor, Tensor], device: torch.device
):
    tensor_a, tensor_b = tensors
    loss = mse_loss_with_chunking(tensor_a, tensor_b, device, max_chunk_memory=1024)
    expected_loss = (
        (tensor_a - tensor_b).float().pow(2).sum() / tensor_a.numel()
    ).item()
    assert pytest.approx(loss) == expected_loss


# Test the reclaim_memory function
@requires_gpu
def test_reclaim_memory_frees_up_memory(device):
    tensor = torch.randn(1000, 1000, device=device)

    initial_memory = torch.cuda.memory_allocated()
    # Delete the tensor and reclaim memory
    reclaim_memory(tensor)
    final_memory = torch.cuda.memory_allocated()

    # Check that memory usage has decreased
    assert final_memory <= initial_memory
