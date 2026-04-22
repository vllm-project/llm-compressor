from dataclasses import dataclass, fields, is_dataclass

import pytest
import torch

from llmcompressor.pipelines.cache import IntermediatesCache, OverrideEqMode


@pytest.mark.unit
def test_new_fetch_update_roundtrip():
    """Test basic fetch/update roundtrip with flat key-value."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    tensor = torch.randn(2, 3)
    key = "test_key"

    cache.update(key, tensor)
    fetched = cache.fetch(key)
    assert torch.equal(fetched, tensor)


@pytest.mark.unit
def test_new_fetch_key_not_found():
    """Test KeyError raised when key not found."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    with pytest.raises(KeyError):
        cache.fetch("nonexistent")


@pytest.mark.unit
def test_new_update_accepts_any_value():
    """Test update accepts non-tensor values."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    cache.update("key", "not a tensor")  # Should not raise
    assert cache.fetch("key") == "not a tensor"


@pytest.mark.unit
def test_new_delete():
    """Test delete removes entry."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    cache.update("key", torch.randn(2, 3))
    assert "key" in cache
    cache.delete("key")
    assert "key" not in cache


@pytest.mark.unit
def test_new_contains():
    """Test __contains__ works correctly."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    tensor = torch.randn(2, 3)
    key = "test_key"
    assert key not in cache
    cache.update(key, tensor)
    assert key in cache


@pytest.mark.unit
def test_new_iter_prefetch():
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
def test_iter_prefetch_grouped():
    """Test iter_prefetch_grouped with (batch_idx, name) keys."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))

    for batch_idx in range(3):
        cache.update((batch_idx, "input_ids"), torch.tensor([batch_idx]))
        cache.update((batch_idx, "attention_mask"), torch.tensor([1]))

    keys = list(cache._store.keys())
    groups = list(cache.iter_prefetch_grouped(keys, group_by=lambda k: k[0]))

    assert len(groups) == 3
    for i, group_dict in enumerate(groups):
        assert (i, "input_ids") in group_dict
        assert (i, "attention_mask") in group_dict
        assert group_dict[(i, "input_ids")].item() == i


@pytest.mark.unit
def test_iter_prefetch_grouped_awq_keys():
    """Test iter_prefetch_grouped with AWQ (module, batch_idx, arg_name) keys."""
    from torch.nn import Linear

    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    module1 = Linear(10, 10)
    module2 = Linear(10, 10)

    for batch_idx in range(2):
        cache.update((module1, batch_idx, "hidden"), torch.tensor([batch_idx]))
        cache.update((module2, batch_idx, "hidden"), torch.tensor([batch_idx + 10]))

    keys = list(cache._store.keys())
    groups = list(cache.iter_prefetch_grouped(keys, group_by=lambda k: (k[0], k[1])))

    assert len(groups) == 4  # 2 modules x 2 batches
    for group_dict in groups:
        assert len(group_dict) == 1
        key = next(iter(group_dict.keys()))
        assert key[2] == "hidden"


@pytest.mark.unit
def test_iter_prefetch_grouped_empty():
    """Test iter_prefetch_grouped with empty keys."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    groups = list(cache.iter_prefetch_grouped([], group_by=lambda k: k))
    assert len(groups) == 0


@pytest.mark.unit
def test_new_iter():
    """Test __iter__ yields all tensors."""
    cache = IntermediatesCache(offload_device=torch.device("cpu"))
    keys = ["a", "b", "c"]
    for k in keys:
        cache.update(k, torch.randn(2, 3))

    iterated = list(cache)
    assert len(iterated) == 3


@pytest.mark.unit
def test_new_clear():
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


@pytest.mark.integration
def test_awq_cache_flow():
    """Test IntermediatesCache works for AWQ parent args flow."""
    from torch.nn import Module, Linear
    import torch

    cache = IntermediatesCache(offload_device=torch.device("cpu"))

    # Simulate AWQ parent args cache behavior
    module = Linear(10, 10)
    batch_idx = 0

    # Store args
    hidden_states = torch.randn(2, 10)
    attention_mask = torch.randn(2, 10)

    cache.update((module, batch_idx, "hidden_states"), hidden_states)
    cache.update((module, batch_idx, "attention_mask"), attention_mask)

    # Fetch args for forward pass
    fetched_kwargs = {
        "hidden_states": cache.fetch((module, batch_idx, "hidden_states")),
        "attention_mask": cache.fetch((module, batch_idx, "attention_mask")),
    }

    assert torch.equal(hidden_states, fetched_kwargs["hidden_states"])
    assert torch.equal(attention_mask, fetched_kwargs["attention_mask"])
