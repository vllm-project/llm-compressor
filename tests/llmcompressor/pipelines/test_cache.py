from dataclasses import dataclass, fields, is_dataclass

import pytest
import torch
from torch.utils.data import DataLoader, StackDataset

from llmcompressor.core import active_session
from llmcompressor.pipelines.cache import (
    IntermediateCache,
    OverrideEqMode,
    build_batches_from_dataloader,
    delete_from_batch,
    fetch_batch,
    iter_batches,
    maybe_prefetch,
    update_batch,
)


@dataclass
class SampleDataclass:
    a: torch.Tensor
    b: int


values_to_test = [
    torch.randn(2, 3).to("cpu"),
    SampleDataclass(a=torch.randn(2, 3), b=42),
    torch.float32,
    [1, 2, 3],
]


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
    return build_batches_from_dataloader(
        dataloader=sample_dataloader,
        model_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )


class TestIntermediateCache:
    """Unit tests for IntermediateCache and related cache functions."""

    @pytest.mark.unit
    def test_offload_and_onload(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        value = torch.randn(2, 3, device="cuda:0")
        cache = IntermediateCache(value, torch.device("cpu"))
        onloaded = cache.fetch()
        assert onloaded.device == torch.device("cuda:0")
        assert torch.equal(onloaded, value)

    @pytest.mark.unit
    def test_onload_device_defaults_to_tensor_device(self):
        """onload_device defaults to the tensor's current device when not specified."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        value = torch.randn(2, 3, device="cuda:0")
        cache = IntermediateCache(value, offload_device=torch.device("cpu"))
        assert cache.onload_device == value.device
        assert cache.value.device == torch.device("cpu")

    @pytest.mark.unit
    def test_offload_none_means_no_op(self):
        """offload_device=None keeps the tensor on its current device."""
        value = torch.randn(2, 3, device="cpu")
        cache = IntermediateCache(
            value, offload_device=None, onload_device=torch.device("cpu")
        )
        # Tensor should remain on its original device
        assert cache.value.device == value.device
        assert cache.value is value

    @pytest.mark.unit
    def test_offload_same_as_onload_no_op(self):
        """offload_device==onload_device keeps the tensor in place."""
        value = torch.randn(2, 3, device="cpu")
        cache = IntermediateCache(
            value, offload_device=torch.device("cpu"), onload_device=torch.device("cpu")
        )
        # Tensor should remain on its original device (no offload needed)
        assert cache.value.device == value.device

    @pytest.mark.unit
    def test_fetch_returns_onload_device(self):
        """fetch() returns the tensor on onload_device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        cuda_device = torch.device("cuda:0")
        value = torch.randn(2, 3, device=cuda_device)
        cache = IntermediateCache(
            value, offload_device=torch.device("cpu"), onload_device=cuda_device
        )
        # Fetch should return tensor on cuda
        fetched = cache.fetch()
        assert fetched.device.type == cuda_device.type
        assert fetched.device.index == cuda_device.index

    @pytest.mark.unit
    def test_update_changes_value(self):
        """update() changes the stored value and offloads it."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        value1 = torch.randn(2, 3, device="cuda:0")
        value2 = torch.randn(2, 3, device="cuda:0")
        cache = IntermediateCache(value1, offload_device=torch.device("cpu"))
        cache.update(value2)
        assert torch.equal(cache.value, value2.to("cpu"))
        assert cache.value.device == torch.device("cpu")

    @pytest.mark.unit
    def test_onload_override_device(self):
        """onload_(device) override moves tensor to specified device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        cuda_device = torch.device("cuda:0")
        cpu_device = torch.device("cpu")
        value = torch.randn(2, 3, device=cuda_device)
        cache = IntermediateCache(
            value, offload_device=cpu_device, onload_device=cuda_device
        )
        # Override onload to cpu
        cache.onload_(cpu_device)
        assert cache.value.device.type == cpu_device.type

    @pytest.mark.unit
    def test_onloaded_context_manager(self):
        """onloaded() context manager moves tensor to specified device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        cuda_device = torch.device("cuda:0")
        cpu_device = torch.device("cpu")
        value = torch.randn(2, 3, device=cuda_device)
        cache = IntermediateCache(
            value, offload_device=cpu_device, onload_device=cuda_device
        )
        # Test onloaded context manager
        with cache.onloaded(cuda_device):
            assert cache.value.device.type == cuda_device.type
        # After context, should return to offload_device
        assert cache.value.device.type == cpu_device.type


class TestBatches:
    """Unit tests for batch-related functions."""

    @pytest.mark.unit
    def test_from_dataloader(self, sample_dataloader):
        cache = build_batches_from_dataloader(sample_dataloader)

        onloaded = fetch_batch(cache[0], ["input_ids"])["input_ids"]
        assert isinstance(onloaded, torch.Tensor)

    @pytest.mark.unit
    @pytest.mark.parametrize("value", values_to_test)
    def test_from_dataloader_diff_types(self, value):
        dataset = StackDataset(value=[value])
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
        cache = build_batches_from_dataloader(dataloader)

        onloaded = fetch_batch(cache[0], ["value"])["value"]
        assert deep_equal(onloaded, value)

    @pytest.mark.unit
    def test_fetch_batch(self, sample_cache):
        fetched = fetch_batch(sample_cache[0], ["input_ids", "attention_mask"])

        assert isinstance(fetched, dict)
        assert "input_ids" in fetched
        assert "attention_mask" in fetched
        assert isinstance(fetched["input_ids"], torch.Tensor)
        assert isinstance(fetched["attention_mask"], torch.Tensor)

    @pytest.mark.unit
    def test_update_batch(self, sample_cache):
        new_outputs = {
            "hidden_states": torch.randn(2, 4, 768),
            "logits": torch.randn(2, 4, 1000),
        }

        update_batch(sample_cache[0], new_outputs, torch.device("cpu"))

        # Verify the updates were stored
        assert "hidden_states" in sample_cache[0]
        assert "logits" in sample_cache[0]

    @pytest.mark.unit
    def test_delete_from_batch(self, sample_cache):
        # First add some intermediates
        new_outputs = {
            "hidden_states": torch.randn(2, 4, 768),
            "logits": torch.randn(2, 4, 1000),
        }
        update_batch(sample_cache[0], new_outputs, torch.device("cpu"))

        # Then delete them
        delete_from_batch(sample_cache[0], ["hidden_states"])

        assert "hidden_states" not in sample_cache[0]
        assert "logits" in sample_cache[0]

    @pytest.mark.unit
    def test_device_handling(self, sample_dataloader):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cuda_device = torch.device("cuda:0")
        cpu_device = torch.device("cpu")

        # Create a cache with GPU as model device and CPU as offload device
        cache = build_batches_from_dataloader(
            dataloader=sample_dataloader,
            model_device=cuda_device,
            offload_device=cpu_device,
        )

        # Add some GPU tensors
        new_outputs = {"hidden_states": torch.randn(2, 3).to(cuda_device)}
        update_batch(cache[0], new_outputs, cpu_device)

        # Verify tensors are offloaded to CPU
        assert cache[0]["hidden_states"].value.device.type == "cpu"

        # Verify tensors are loaded back to GPU when fetched
        fetched = fetch_batch(cache[0], ["hidden_states"])
        assert fetched["hidden_states"].device.type == "cuda"

    @pytest.mark.unit
    def test_iter_batches_basic(self, sample_cache):
        """Test iter_batches yields correct batch structure."""
        batches = list(iter_batches(sample_cache))

        assert len(batches) == len(sample_cache)
        for batch in batches:
            assert isinstance(batch, dict)
            assert "input_ids" in batch
            assert "attention_mask" in batch

    @pytest.mark.unit
    def test_iter_batches_with_input_names(self, sample_cache):
        """Test iter_batches with specific input_names filter."""
        batches = list(iter_batches(sample_cache, input_names=["input_ids"]))

        assert len(batches) == len(sample_cache)
        for batch in batches:
            assert list(batch.keys()) == ["input_ids"]


class TestMaybePrefetch:
    """Unit tests for maybe_prefetch function."""

    @pytest.mark.unit
    def test_maybe_prefetch_matches_iter(self, sample_cache):
        """maybe_prefetch yields the same batch contents as iter."""
        via_iter = list(iter_batches(sample_cache))
        via_prefetch = list(maybe_prefetch(sample_cache))
        assert len(via_iter) == len(via_prefetch)
        for i, (b_iter, b_prefetch) in enumerate(zip(via_iter, via_prefetch)):
            assert deep_equal(b_iter, b_prefetch), f"batch {i} differs"

    @pytest.mark.unit
    def test_maybe_prefetch_use_prefetch(self, sample_cache):
        """Test maybe_prefetch actually prefetches."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        session = active_session()
        session.state.sequential_prefetch = True

        result = list(maybe_prefetch(sample_cache))
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "input_ids" in result[0]
        assert torch.equal(
            result[0]["input_ids"],
            torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]], dtype=torch.long),
        )

    @pytest.mark.unit
    def test_maybe_prefetch_no_cuda(self):
        """Test maybe_prefetch works correctly without CUDA."""
        cache = build_batches_from_dataloader(
            dataloader=DataLoader(StackDataset(input_ids=torch.tensor([[1, 2]]))),
            model_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
        )

        result = list(maybe_prefetch(cache))
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "input_ids" in result[0]
        assert torch.equal(result[0]["input_ids"], torch.tensor([[1, 2]]))


def deep_equal(a, b) -> bool:
    if type(a) is not type(b):
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
            a_dict = {field.name: getattr(a, field.name) for field in fields(a)}
            b_dict = {field.name: getattr(b, field.name) for field in fields(b)}

            return deep_equal(a_dict, b_dict)
        case _:
            return a == b


def test_override_eq_mode():
    a = torch.tensor([1, 2, 3])
    b = a
    c = torch.tensor([2, 2, 2])

    with pytest.raises(RuntimeError):
        assert a == b
    with pytest.raises(RuntimeError):
        assert not (a == c)

    with OverrideEqMode():
        assert a == b
        assert not (a == c)
