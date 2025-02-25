from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import DataLoader, StackDataset

from llmcompressor.pipelines.cache import IntermediatesCache, IntermediateValue


@pytest.fixture
def sample_dataloader():
    # Create sample input tensors
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)

    # Create dataset and dataloader
    dataset = StackDataset(input_ids=input_ids, attention_mask=attention_mask)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def sample_cache(sample_dataloader):
    return IntermediatesCache.from_dataloader(
        dataloader=sample_dataloader,
        model_device=torch.device("cpu"),
        mask_padding=True,
        offload_device=torch.device("cpu"),
    )


@pytest.mark.unit
def test_initialization(sample_dataloader):
    cache = IntermediatesCache.from_dataloader(
        dataloader=sample_dataloader,
        model_device=torch.device("cpu"),
        mask_padding=True,
    )

    assert isinstance(cache, IntermediatesCache)
    assert len(cache.batch_intermediates) > 0
    assert isinstance(cache.batch_intermediates[0], dict)


@pytest.mark.unit
def test_fetch_inputs(sample_cache):
    fetched = sample_cache.fetch(0, ["input_ids", "attention_mask"])

    assert isinstance(fetched, dict)
    assert "input_ids" in fetched
    assert "attention_mask" in fetched
    assert isinstance(fetched["input_ids"], torch.Tensor)
    assert isinstance(fetched["attention_mask"], torch.Tensor)


@pytest.mark.unit
def test_update_intermediates(sample_cache):
    new_outputs = {
        "hidden_states": torch.randn(2, 4, 768),
        "logits": torch.randn(2, 4, 1000),
    }

    sample_cache.update(0, new_outputs)

    # Verify the updates were stored
    assert "hidden_states" in sample_cache.batch_intermediates[0]
    assert "logits" in sample_cache.batch_intermediates[0]


@pytest.mark.unit
def test_delete_intermediates(sample_cache):
    # First add some intermediates
    new_outputs = {
        "hidden_states": torch.randn(2, 4, 768),
        "logits": torch.randn(2, 4, 1000),
    }
    sample_cache.update(0, new_outputs)

    # Then delete them
    sample_cache.delete(0, ["hidden_states"])

    assert "hidden_states" not in sample_cache.batch_intermediates[0]
    assert "logits" in sample_cache.batch_intermediates[0]


@pytest.mark.unit
def test_mask_padding():
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])

    masked = IntermediatesCache._mask_padding(input_ids, attention_mask)

    # Check if padding tokens are properly masked
    expected = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
    assert torch.equal(masked, expected)


@pytest.mark.unit
def test_offload_and_onload_tensor():
    cache = IntermediatesCache([], torch.device("cpu"))

    # Test tensor offloading
    original_tensor = torch.randn(2, 3).to("cpu")
    offloaded = cache._offload_value(original_tensor)

    assert isinstance(offloaded, IntermediateValue)
    assert isinstance(offloaded.value, torch.Tensor)
    assert offloaded.device == original_tensor.device

    # Test tensor onloading
    onloaded = cache._onload_value(offloaded)
    assert torch.equal(onloaded, original_tensor)


@dataclass
class SampleDataclass:
    a: torch.Tensor
    b: int


@pytest.mark.unit
def test_offload_and_onload_dataclass():
    cache = IntermediatesCache([], torch.device("cpu"))

    # Create a sample dataclass instance
    sample_data = SampleDataclass(a=torch.randn(2, 3), b=42)

    # Test dataclass offloading
    offloaded = cache._offload_value(sample_data)
    assert isinstance(offloaded, IntermediateValue)
    assert isinstance(offloaded.value, SampleDataclass)
    assert isinstance(offloaded.value.a, IntermediateValue)
    assert isinstance(offloaded.value.b, IntermediateValue)

    # Test dataclass onloading
    onloaded = cache._onload_value(offloaded)
    assert onloaded == sample_data


@pytest.mark.unit
def test_offload_and_onload_dtype():
    cache = IntermediatesCache([], torch.device("cpu"))

    # Create a sample dataclass instance
    sample_data = torch.float32

    # Test dataclass offloading
    offloaded = cache._offload_value(sample_data)
    assert isinstance(offloaded, IntermediateValue)
    assert isinstance(offloaded.value, torch.dtype)

    # Test dataclass onloading
    onloaded = cache._onload_value(offloaded)
    assert onloaded == sample_data


@pytest.mark.unit
def test_4d_attention_mask():
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.ones(1, 1, 1, 4)  # 4D attention mask

    masked = IntermediatesCache._mask_padding(input_ids, attention_mask)

    # Check if the function handles 4D attention mask properly
    expected = torch.tensor([[1, 2, 3, 0]])
    assert torch.equal(masked, expected)


@pytest.mark.unit
def test_device_handling(sample_dataloader):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cuda_device = torch.device("cuda")
    cpu_device = torch.device("cpu")

    # Create a cache with GPU as model device and CPU as offload device
    cache = IntermediatesCache.from_dataloader(
        dataloader=sample_dataloader,
        model_device=cuda_device,
        offload_device=cpu_device,
    )

    # Add some GPU tensors
    new_outputs = {"hidden_states": torch.randn(2, 3).to(cuda_device)}
    cache.update(0, new_outputs)

    # Verify tensors are offloaded to CPU
    assert cache.batch_intermediates[0]["hidden_states"].value.device.type == "cpu"

    # Verify tensors are loaded back to GPU when fetched
    fetched = cache.fetch(0, ["hidden_states"])
    assert fetched["hidden_states"].device.type == "cuda"
