from typing import Optional

from dataclasses import dataclass, fields, is_dataclass

import pytest
import torch
from torch.utils.data import DataLoader, StackDataset

from llmcompressor.pipelines.cache import IntermediatesCache


@dataclass(frozen=True)
class SampleDataclass:
    a: int
    b: Optional[torch.Tensor] = None 
    c: Optional["SampleDataclass"] = None


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


values_to_test = [
    torch.randn(2, 3).to("cpu"),
    SampleDataclass(a=42, b=torch.randn(2, 3), c=SampleDataclass(a=64)),
    torch.float32,
    [1, 2, 3],
]


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
@pytest.mark.parametrize("value", values_to_test)
def test_from_dataloader(value):
    dataset = StackDataset(value=[value])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    cache = IntermediatesCache.from_dataloader(dataloader)

    onloaded = cache.fetch(0, ["value"])["value"]
    assert deep_equal(onloaded, value)


@pytest.mark.unit
@pytest.mark.parametrize("value", values_to_test)
def test_offload_and_onload(value):
    offloaded = IntermediatesCache._offload_value(value, torch.device("cpu"))
    onloaded = IntermediatesCache._onload_value(offloaded)
    assert deep_equal(onloaded, value)


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


def deep_equal(a, b) -> bool:
    if type(a) != type(b):
        return False

    match a:
        case torch.Tensor():
            return torch.equal(a, b)
        case list() | tuple():
            if len(a) != len(b):
                return False
            return all(deep_equal(_a, _b) for _a, _b in zip(a, b))
        case dict():
            if a.keys() != b.keys():
                return False
            return all(deep_equal(a[key], b[key]) for key in a.keys())
        case _ if is_dataclass(a):
            a_dict = {field: getattr(a, field.name) for field in fields(a)}
            b_dict = {field: getattr(b, field.name) for field in fields(b)}

            return deep_equal(a_dict, b_dict)
        case _:
            return a == b
