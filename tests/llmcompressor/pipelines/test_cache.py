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
    """Test that iter_prefetch overlaps onload with processing, reducing total time.

    Simulates realistic scenario:
    - onload: time to transfer data (e.g., H2D transfer)
    - processing: time to use data in main thread (e.g., forward pass)

    With prefetch: while main thread processes item N, background thread loads item N+1.

    Using large delays (50ms) to make threading overhead negligible, so we can
    measure actual speedup from overlap.
    """
    cache = IntermediatesCache(offload_device=torch.device("cpu"))

    num_items = 10
    onload_delay = 0.05  # 50ms - simulate data transfer
    processing_delay = 0.05  # 50ms - simulate forward pass

    # Store items
    for i in range(num_items):
        cache.update(i, torch.randn(10, 10))

    onload_thread_ids = []

    original_onload = IntermediatesCache._onload_value

    def delayed_onload(cls, value):
        onload_thread_ids.append(threading.current_thread().ident)
        time.sleep(onload_delay)  # Simulate H2D transfer time
        return original_onload(value)

    def simulate_processing(data):
        time.sleep(processing_delay)  # Simulate forward pass time
        return data

    # Test 1: Sequential (no prefetch)
    with patch.object(IntermediatesCache, "_onload_value", delayed_onload):
        onload_thread_ids.clear()
        start_sequential = time.time()
        for item in cache.iter(list(range(num_items))):
            simulate_processing(item)
        sequential_time = time.time() - start_sequential

    # Test 2: Prefetch (overlap)
    with patch.object(IntermediatesCache, "_onload_value", delayed_onload):
        onload_thread_ids.clear()
        start_prefetch = time.time()
        for item in cache.iter_prefetch(list(range(num_items))):
            simulate_processing(item)
        prefetch_time = time.time() - start_prefetch
        prefetch_threads = set(onload_thread_ids)

    # Verify: onload calls happened in multiple threads (background prefetch works)
    assert (
        len(prefetch_threads) > 1
    ), f"Expected multiple threads for prefetch, got threads: {prefetch_threads}"

    # Verify timing:
    # Sequential: num_items * (onload + processing) = 10 * (0.05 + 0.05) = 1.0s
    expected_sequential = num_items * (onload_delay + processing_delay)
    assert (
        sequential_time >= expected_sequential * 0.85
    ), f"Sequential time {sequential_time:.3f}s should be ~{expected_sequential:.3f}s"

    # Prefetch with perfect overlap:
    # - First onload (0.05s) + first processing starts
    # - While processing N, onload N+1 happens in background
    # - Total ≈ onload_delay + num_items * max(onload, processing) + small overhead
    # When onload == processing: ≈ (num_items + 1) * delay = 0.55s (vs 1.0s sequential)
    # Expected speedup: ~1.8x
    # Prefetch should be at least 70% faster than sequential (speedup >= 1.7x)
    # With large delays, threading overhead becomes negligible
    speedup = sequential_time / prefetch_time
    assert speedup >= 1.7, (
        f"Prefetch should provide significant speedup. "
        f"Sequential: {sequential_time:.3f}s, Prefetch: {prefetch_time:.3f}s, "
        f"Speedup: {speedup:.2f}x (expected >= 1.7x)"
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
