from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace
from typing import Optional
from weakref import ref

import pytest
import torch
from torch.utils.data import DataLoader, StackDataset

from llmcompressor.pipelines.cache import (
    IntermediatesCache,
    IntermediateValue,
    _supports_record_stream,
)


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
    )

    assert isinstance(cache, IntermediatesCache)
    assert len(cache.batch_intermediates) > 0
    assert isinstance(cache.batch_intermediates[0], dict)


@pytest.mark.unit
@pytest.mark.parametrize("offload_device", ["cpu", torch.device("cpu"), None])
def test_cache_normalizes_device_arguments(sample_dataloader, offload_device):
    expected_device = None if offload_device is None else torch.device("cpu")
    cache = IntermediatesCache(offload_device=offload_device)
    empty_cache = IntermediatesCache.empty(1, offload_device)
    dataloader_cache = IntermediatesCache.from_dataloader(
        sample_dataloader, model_device="cpu", offload_device=offload_device
    )

    for instance in (cache, empty_cache, dataloader_cache):
        assert instance.offload_device == expected_device

    for intermediate in dataloader_cache.batch_intermediates[0].values():
        assert intermediate.device == torch.device("cpu")

    assert IntermediatesCache().offload_device == torch.device("cpu")


@pytest.mark.unit
def test_iter_prefetch_empty_cache():
    """iter_prefetch yields nothing when cache has no batches."""
    cache = IntermediatesCache.empty(0, torch.device("cpu"))
    assert list(cache.iter_prefetch()) == []


@pytest.mark.unit
def test_iter_prefetch_matches_iter(sample_cache):
    """iter_prefetch yields the same batch contents as iter."""

    def batch_dicts_equal(a: dict, b: dict) -> bool:
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k]) for k in a)

    via_iter = list(sample_cache.iter())
    via_prefetch = list(sample_cache.iter_prefetch())
    assert len(via_iter) == len(via_prefetch)
    for i, (b_iter, b_prefetch) in enumerate(zip(via_iter, via_prefetch)):
        assert batch_dicts_equal(b_iter, b_prefetch), f"batch {i} differs"


@pytest.mark.unit
def test_record_stream_skips_unsupported_backend(monkeypatch):
    """_record_stream must not call record_stream on backends where it is
    unimplemented (e.g. MPS), otherwise iter_prefetch crashes."""
    monkeypatch.setattr(
        "llmcompressor.pipelines.cache._supports_record_stream", lambda device: False
    )

    def fail(self, stream):
        raise AssertionError("record_stream must not be called")

    monkeypatch.setattr(torch.Tensor, "record_stream", fail)

    tensor = torch.zeros(1)
    stream = SimpleNamespace(device=tensor.device)
    # dict branch exercised as well; must complete without calling record_stream
    IntermediatesCache._record_stream({"hidden": [tensor]}, stream)


@pytest.mark.unit
def test_supports_record_stream_cpu_unsupported():
    _supports_record_stream.cache_clear()
    assert _supports_record_stream(torch.device("cpu")) is False


@pytest.mark.unit
def test_iter_prefetch_onloads_accelerator_tensors(monkeypatch):
    """Regression test: prefetching tensors which onload to a device without
    record_stream support (e.g. MPS) must not crash."""
    accel = torch.accelerator.current_accelerator()
    if accel is None:
        pytest.skip("No accelerator available")

    monkeypatch.setattr(
        "llmcompressor.pipelines.cache._supports_record_stream", lambda device: False
    )
    cache = IntermediatesCache.empty(2, torch.device("cpu"))
    for i in range(2):
        cache.update(i, {"hidden_states": torch.randn(2, 3, device=accel)})

    batches = list(cache.iter_prefetch())
    assert len(batches) == 2
    assert batches[0]["hidden_states"].device.type == accel.type


@pytest.mark.unit
def test_iter_prefetch_non_cuda_onload_stays_on_caller_thread(monkeypatch):
    """Batches onloading to accelerators without thread-safe op submission
    (e.g. MPS) must be fetched synchronously: submitting device ops from the
    prefetch worker thread can abort the process."""
    # flat and nested (device=None container) non-CUDA targets
    intermediates = [
        IntermediateValue(torch.randn(2, 3), torch.device("mps")),
        IntermediateValue(
            value=[IntermediateValue(torch.randn(2, 3), torch.device("mps"))],
            device=None,
        ),
    ]

    def fail_on_thread(*args, **kwargs):
        raise AssertionError("prefetch worker thread must not be used")

    monkeypatch.setattr(
        "llmcompressor.pipelines.cache.ThreadPoolExecutor", fail_on_thread
    )
    monkeypatch.setattr(
        IntermediatesCache, "fetch", lambda self, idx, names=None: {"i": idx}
    )

    for intermediate in intermediates:
        cache = IntermediatesCache.empty(1, torch.device("cpu"))
        cache.batch_intermediates[0]["hidden_states"] = intermediate
        assert list(cache.iter_prefetch()) == [{"i": 0}]
        assert cache._prefetch_stream is None


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
    cache = IntermediatesCache()
    offloaded = cache._offload_value(value, torch.device("cpu"))
    onloaded = cache._onload_value(offloaded)
    assert deep_equal(onloaded, value)


@pytest.mark.unit
def test_shared_offload_uses_identity():
    cache = IntermediatesCache.empty(1, torch.device("meta"))
    tensor = torch.ones(2, 3)
    cache.update(0, {"shared": [tensor, tensor], "distinct": tensor.clone()})

    batch = cache.batch_intermediates[0]
    first, second = batch["shared"].value
    assert first.value.device.type == "meta"
    assert first.value is second.value
    assert first.value is not batch["distinct"].value


@pytest.mark.unit
def test_shared_offload_respects_each_cache_device():
    source = torch.ones(2, 3)
    meta_cache = IntermediatesCache.empty(1, "meta")
    cpu_cache = IntermediatesCache.empty(1, torch.device("cpu"))
    no_offload_cache = IntermediatesCache.empty(1, None)

    for cache in (meta_cache, cpu_cache, no_offload_cache):
        cache.update(0, {"source": source})

    offloaded = meta_cache.batch_intermediates[0]["source"].value
    assert offloaded.device.type == "meta"
    assert cpu_cache.fetch(0)["source"] is source
    assert no_offload_cache.fetch(0)["source"] is source
    assert (
        IntermediatesCache._offload_value(source, torch.device("meta")).value
        is offloaded
    )


@pytest.mark.unit
def test_unindexed_offload_tracks_current_accelerator_device(monkeypatch):
    current_index = 0
    transfers = []

    class SourceTensor(torch.Tensor):
        def to(self, *, device):
            transfers.append(device)
            return self.as_subclass(torch.Tensor).clone()

    source = torch.ones(2, 3).as_subclass(SourceTensor)
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda: torch.device("cuda")
    )
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda: current_index
    )

    cache = IntermediatesCache.empty(1, "cuda")
    cache.update(0, {"source": source})
    first = cache.batch_intermediates[0]["source"].value
    current_index = 1
    cache.update(0, {"source": source})
    second = cache.batch_intermediates[0]["source"].value
    again = IntermediatesCache._offload_value(source, torch.device("cuda:0")).value

    assert cache.offload_device == torch.device("cuda")
    assert first is not second
    assert first is again
    assert transfers == [torch.device("cuda:0"), torch.device("cuda:1")]


@pytest.mark.unit
def test_offload_entry_expires_with_source():
    source = torch.ones(2, 3)
    meta = torch.device("meta")
    cpu = torch.device("cpu:0")
    keys = {(id(source), meta), (id(source), cpu)}
    source_ref = ref(source)
    offloaded = IntermediatesCache._offload_value(source, meta)
    IntermediatesCache._offload_value(source, cpu)

    assert keys <= IntermediatesCache.offload_values.keys()
    del source
    assert source_ref() is None
    assert keys.isdisjoint(IntermediatesCache.offload_values)
    assert offloaded.value.device.type == "meta"


@pytest.mark.unit
def test_offload_replaces_stale_identity_entry(monkeypatch):
    old_source = torch.ones(2, 3)
    old_ref = ref(old_source)
    old_offload = old_source.to("meta")
    del old_source
    assert old_ref() is None

    source = torch.ones(4, 5)
    device = torch.device("meta")
    key = (id(source), device)
    # Simulate id reuse without depending on Python's allocator.
    entries = {key: (old_ref, old_offload)}
    monkeypatch.setattr(IntermediatesCache, "offload_values", entries)

    offloaded = IntermediatesCache._offload_value(source, device)

    assert offloaded.value.shape == source.shape
    assert entries[key][0]() is source
    assert entries[key][1] is offloaded.value


@pytest.mark.unit
def test_noop_offload_does_not_retain_source():
    source = torch.ones(2, 3)
    source_ref = ref(source)
    device = torch.device("cpu")
    offloaded = IntermediatesCache._offload_value(source, device)

    assert offloaded.value is source
    assert (id(source), device) not in IntermediatesCache.offload_values
    del offloaded, source
    assert source_ref() is None


@pytest.mark.unit
def test_device_handling(sample_dataloader):
    if not torch.accelerator.is_available():
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
            a_dict = {field: getattr(a, field.name) for field in fields(a)}
            b_dict = {field: getattr(b, field.name) for field in fields(b)}

            return deep_equal(a_dict, b_dict)
        case _:
            return a == b
