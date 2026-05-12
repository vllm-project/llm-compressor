import threading
import time
from dataclasses import fields, is_dataclass
from unittest.mock import patch

import pytest
import torch

from llmcompressor.pipelines.cache import IntermediatesCache, OverrideEqMode


@pytest.mark.unit
def test_fetch_update_roundtrip():
    """Test basic fetch/update roundtrip with flat key-value."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    tensor = torch.randn(2, 3)
    key = "test_key"

    cache.update(key, tensor)
    fetched = cache[key]
    assert torch.equal(fetched, tensor)


@pytest.mark.unit
def test_fetch_key_not_found():
    """Test KeyError raised when key not found."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    with pytest.raises(KeyError):
        _ = cache["nonexistent"]


@pytest.mark.unit
def test_update_accepts_any_value():
    """Test update accepts non-tensor values."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    cache.update("key", "not a tensor")  # Should not raise
    assert cache["key"] == "not a tensor"


@pytest.mark.unit
def test_append():
    """Test append adds to existing list."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    cache.append("key", [1, 2])
    cache.append("key", [3, 4])
    assert cache[("key", 0)] == [1, 2]
    assert cache[("key", 1)] == [3, 4]


@pytest.mark.unit
def test_append_no_prefix():
    """Test append with no key_prefix uses pure integer keys."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    key0, idx0 = cache.append(None, "aaa")
    key1, idx1 = cache.append(None, "bbb")
    assert key0 == 0 and idx0 == 0
    assert key1 == 1 and idx1 == 1
    assert cache[0] == "aaa"
    assert cache[1] == "bbb"


@pytest.mark.unit
def test_delete():
    """Test delete removes entry."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    cache.update("key", torch.randn(2, 3))
    assert "key" in cache
    cache.delete("key")
    assert "key" not in cache


@pytest.mark.unit
def test_contains():
    """Test __contains__ works correctly."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    tensor = torch.randn(2, 3)
    key = "test_key"
    assert key not in cache
    cache.update(key, tensor)
    assert key in cache


@pytest.mark.unit
def test_iter_prefetch():
    """Test iter_prefetch with explicit key list."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    keys = ["a", "b", "c"]
    tensors = [torch.randn(2, 3) for _ in keys]

    for k, t in zip(keys, tensors):
        cache.update(k, t)

    prefetched = list(cache.iter_prefetch(keys))
    assert len(prefetched) == 3
    for original, prefetched_tensor in zip(tensors, prefetched):
        assert torch.equal(original, prefetched_tensor)


@pytest.mark.unit
def test_iter_prefetch_with_prefix():
    """Test iter_prefetch with prefix pattern."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))

    # Store batch dicts like sequential pipeline with module prefix
    for batch_idx in range(3):
        batch_dict = {
            "input_ids": torch.tensor([batch_idx]),
            "attention_mask": torch.tensor([1]),
        }
        cache.update(("module", batch_idx), batch_dict)

    # Use prefix "module" to match all (module, 0), (module, 1), etc.
    batches = list(cache.iter_prefetch(["module"]))
    assert len(batches) == 3
    for i, batch in enumerate(batches):
        assert "input_ids" in batch
        assert batch["input_ids"].item() == i


@pytest.mark.unit
def test_iter_prefetch_runs_in_separate_thread():
    """Test that iter_prefetch runs onload in a background thread."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))

    # Store several items
    for i in range(5):
        cache.update(i, torch.randn(10, 10))

    main_thread_id = threading.current_thread().ident
    onload_thread_ids = []

    original_onload = IntermediatesCache._onload_value

    def tracking_onload(cls, value):
        onload_thread_ids.append(threading.current_thread().ident)
        return original_onload(value)

    with patch.object(IntermediatesCache, "_onload_value", tracking_onload):
        keys = list(range(5))
        results = list(cache.iter_prefetch(keys))

        # Verify all values fetched correctly
        assert len(results) == 5

        # Verify that onload happened in different threads (ThreadPoolExecutor)
        unique_threads = set(onload_thread_ids)
        assert len(unique_threads) == 2, (
            f"Expected onload to run in main and background threads, "
            f"but all ran in thread {unique_threads}"
        )

        # The first onload may be in main thread, but subsequent ones should be
        # prefetched in background thread
        assert (
            main_thread_id in onload_thread_ids
        ), "First item should load in main thread"
        assert any(
            tid != main_thread_id for tid in onload_thread_ids
        ), "Some items should be prefetched in background thread"


@pytest.mark.unit
def test_iter_prefetch_overlaps_onload_operations():
    """Test that iter_prefetch overlaps onload with processing (background prefetch).

    Uses synchronization primitives to deterministically detect concurrent execution:
    if a background thread runs onload while the main thread is processing,
    overlap is proven without relying on wall-clock timing.
    """
    cache = IntermediatesCache(offload_device=torch.device("cpu"))

    num_items = 5
    for i in range(num_items):
        cache.update(i, torch.randn(10, 10))

    processing_active = threading.Event()
    overlap_detected = threading.Event()

    original_onload = IntermediatesCache._onload_value

    def instrumented_onload(cls, value):
        if (
            processing_active.is_set()
            and threading.current_thread() is not threading.main_thread()
        ):
            overlap_detected.set()
        return original_onload(value)

    with patch.object(IntermediatesCache, "_onload_value", instrumented_onload):
        for item in cache.iter_prefetch(list(range(num_items))):
            processing_active.set()
            time.sleep(0.02)
            processing_active.clear()

    assert overlap_detected.is_set(), (
        "iter_prefetch should overlap background onload with main-thread processing, "
        "but no concurrent execution was detected"
    )


@pytest.mark.unit
def test_iter():
    """Test __iter__ yields all tensors."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    keys = ["a", "b", "c"]
    for k in keys:
        cache.update(k, torch.randn(2, 3))

    iterated = list(cache)
    assert len(iterated) == 3


@pytest.mark.unit
def test_clear():
    """Test clear empties cache."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    cache.update("key", torch.randn(2, 3))
    cache.clear()
    assert len(cache) == 0
    assert "key" not in cache


@pytest.mark.unit
def test_offload_and_onload():
    """Test _offload_value and _onload_value preserve tensor values."""
    value = torch.randn(2, 3)
    offloaded = IntermediatesCache._offload_value(value, torch.device("cpu"))
    onloaded = IntermediatesCache._onload_value(offloaded)
    assert torch.equal(onloaded, value)


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


@pytest.mark.unit
def test_fetch_no_onload_returns_unwrapped_value_on_offload_device():
    """Test fetch_no_onload returns value without IntermediateValue wrappers on offload device."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    tensor = torch.randn(2, 3)
    cache.update("key", tensor)

    fetched = cache.fetch_no_onload("key")
    assert isinstance(fetched, torch.Tensor)
    assert torch.equal(fetched, tensor)
    assert fetched.device.type == "cpu"


@pytest.mark.unit
def test_fetch_no_onload_nested_structure():
    """Test fetch_no_onload unwraps IntermediateValue in nested dicts."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    value = {
        "a": torch.randn(2, 3),
        "b": torch.randn(4, 5),
        "c": "string_value",
    }
    cache.update("batch", value)

    fetched = cache.fetch_no_onload("batch")
    assert isinstance(fetched, dict)
    assert set(fetched.keys()) == {"a", "b", "c"}
    assert torch.equal(fetched["a"], value["a"])
    assert torch.equal(fetched["b"], value["b"])
    assert fetched["c"] == "string_value"
    assert fetched["a"].device.type == "cpu"
    assert fetched["b"].device.type == "cpu"


@pytest.mark.unit
def test_fetch_no_onload_vs_regular_fetch():
    """Test fetch_no_onload differs from regular fetch by not onloading."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    tensor = torch.randn(2, 3)
    cache.update("key", tensor)

    no_onload = cache.fetch_no_onload("key")
    onloaded = cache["key"]

    assert torch.equal(no_onload, onloaded)
    assert no_onload.device.type == "cpu"


@pytest.mark.unit
def test_prefix_counters_and_indices():
    """Test _prefix_counters and _prefix_indices are maintained correctly."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))

    # update with integer keys
    cache.update(0, "a")
    cache.update(2, "b")
    cache.update(5, "c")
    assert cache._prefix_counters[()] == 6
    assert sorted(cache._prefix_indices[()]) == [0, 2, 5]

    # append uses the counter to generate next index
    _, idx = cache.append(None, "d")
    assert idx == 6
    assert cache._prefix_counters[()] == 7

    # append with tuple prefix
    _, idx0 = cache.append("layer", "x")
    _, idx1 = cache.append("layer", "y")
    assert idx0 == 0 and idx1 == 1
    assert cache._prefix_counters[("layer",)] == 2
    assert cache._prefix_indices[("layer",)] == [0, 1]

    # delete removes from both structures
    cache.delete(2)
    assert 2 not in cache
    assert sorted(cache._prefix_indices[()]) == [0, 5, 6]
    assert cache._prefix_counters[()] == 7

    # delete max index should decrement counter
    cache.delete(("layer", 1))
    assert cache._prefix_counters[("layer",)] == 1

    # iter_keys uses prefix_indices for fast sorted lookup
    cache.update(("module", 3), "a")
    cache.update(("module", 1), "b")
    keys = cache.iter_keys("module")
    assert keys == [("module", 1), ("module", 3)]

    # multiple prefixes tracked independently
    assert cache._prefix_counters[()] == 7
    assert cache._prefix_counters[("layer",)] == 1


@pytest.mark.unit
def test_delete_nonexistent_key_no_error():
    """Test deleting a key that doesn't exist does not raise."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    cache.delete("nonexistent")
    assert len(cache) == 0


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
