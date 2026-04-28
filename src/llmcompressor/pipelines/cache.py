from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generator, Generic, TypeVar
from weakref import WeakKeyDictionary

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils._python_dispatch import TorchDispatchMode
from tqdm import tqdm


TKey = TypeVar("TKey", bound=Any)


@dataclass
class IntermediateValue:
    """
    Wrapper for offloaded tensors, tracking the device to onload to.

    Only used at leaf tensor positions in pytrees stored in the cache.
    """

    value: torch.Tensor
    device: torch.device | None


class IntermediatesCache(Generic[TKey]):
    """
    Cache which stores intermediate values (activations) produced by batched, sequential
    execution of models. Values are offloaded to the `offload_device` when stored in
    the cache and onloaded to their original device when fetched from the cache. If
    `offload_device` is None, values will not be offloaded at all.

    Uses pytree internally to handle nested containers (dicts, tuples, lists, dataclasses).
    IntermediateValue wrappers are placed at tensor leaf positions only.

    Construct using `from_dataloader` class method or directly with offload_device
    """

    _store: dict[TKey, Any]
    offload_device: torch.device | None

    # map of onload value -> offload value
    # used to avoid excess memory usage when shared tensors are offloaded
    offload_values: WeakKeyDictionary[torch.Tensor, torch.Tensor] = WeakKeyDictionary()

    def __init__(self, offload_device: torch.device | None = None):
        self._store = {}
        self.offload_device = offload_device

    @classmethod
    def from_dataloader(
        cls,
        dataloader: torch.utils.data.DataLoader,
        model_device: torch.device = torch.device("cpu"),
        offload_device: torch.device | None = torch.device("cpu"),
    ):
        """
        Initialize a cache with data from the provided dataloader.
        Stores each batch as a dict under key batch_idx.

        :param dataloader: dataloader which generates values to be cached
        :param model_device: device which values will be onloaded to when fetched
        :param offload_device: device to offload values to
        """
        cache = cls(offload_device)
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Preparing cache")):
            cache._store[batch_idx] = cls._offload_value(
                batch, offload_device, model_device
            )
        return cache

    def __getitem__(self, key: TKey) -> Any:
        """
        Fetch a value by key, onloading it to the original device

        :param key: key to fetch
        :return: onloaded value
        :raises KeyError: if key is not found in cache
        """
        if key not in self._store:
            raise KeyError(f"Key {key} not found in cache")
        return self._onload_value(self._store[key])

    def update(self, key: TKey, value: Any) -> None:
        """
        Update/put a value for a key, offloading any tensors

        :param key: key to store value under
        :param value: value to store (tensors will be offloaded)
        """
        self._store[key] = self._offload_value(value, self.offload_device)

    def append(self, key_prefix: TKey | None, value: Any) -> tuple[TKey, int]:
        """
        Append a value with an auto-generated index.

        If key_prefix is None, uses integer keys directly (0, 1, 2, ...).
        Otherwise, creates keys as (key_prefix, index).

        :param key_prefix: key prefix for storing the value, or None for pure integer keys
        :param value: value to store
        :return: tuple of (full_key, index) where full_key is the complete key used
        """
        if key_prefix is None:
            matching_keys = [k for k in self._store.keys() if isinstance(k, int)]
            next_idx = max(matching_keys, default=-1) + 1
            full_key = next_idx
        else:
            if isinstance(key_prefix, tuple):
                prefix = key_prefix
            else:
                prefix = (key_prefix,)

            matching_keys = [
                k for k in self._store.keys()
                if isinstance(k, tuple)
                and len(k) == len(prefix) + 1
                and k[:-1] == prefix
                and isinstance(k[-1], int)
            ]

            next_idx = max((k[-1] for k in matching_keys), default=-1) + 1
            full_key = prefix + (next_idx,)

        self._store[full_key] = self._offload_value(value, self.offload_device)
        return full_key, next_idx

    def delete(self, key: TKey) -> None:
        """
        Delete a value from the cache

        :param key: key to delete
        """
        del self._store[key]

    def __contains__(self, key: TKey) -> bool:
        """
        Check if a key is in the cache

        :param key: key to check
        :return: True if key is in cache, False otherwise
        """
        return key in self._store

    def iter_keys(self, prefix: TKey | None = None) -> list[TKey]:
        """
        Get keys matching a prefix.

        For prefix like (module,), returns all keys (module, 0), (module, 1), etc.
        Non-tuple prefixes are treated as single-element tuples.

        :param prefix: prefix to match keys starting with this prefix
        :return: list of matching keys sorted by final integer index
        """
        if prefix is None:
            return list(self._store.keys())

        if isinstance(prefix, tuple):
            p = prefix
        else:
            p = (prefix,)

        keys = []
        for k in self._store.keys():
            if not isinstance(k, tuple):
                continue
            if len(k) == len(p) + 1 and k[:-1] == p and isinstance(k[-1], int):
                keys.append(k)

        return sorted(keys, key=lambda k: k[-1])

    def iter(self, keys: list[TKey] | None = None) -> Generator[Any, None, None]:
        """
        Iterate over keys with values onloaded in the current thread.

        Keys that are not exact matches are treated as prefixes.
        For example, [module] matches (module, 0), (module, 1), etc.

        :param keys: list of keys or prefixes to iterate over. If None, iterates over all keys.
        """
        if keys is None:
            keys = list(self._store.keys())
        else:
            resolved_keys = []
            for key in keys:
                if key in self._store:
                    resolved_keys.append(key)
                else:
                    resolved_keys.extend(self.iter_keys(key))
            keys = resolved_keys

        for key in keys:
            yield self._onload_value(self._store[key])

    def iter_prefetch(
        self, keys: list[TKey] | None = None
    ) -> Generator[Any, None, None]:
        """
        Iterate over keys with values prefetched in a background thread.
        Overlaps onload from offload_device with consumption of the current value.

        Keys that are not exact matches are treated as prefixes.
        For example, [module] matches (module, 0), (module, 1), etc.

        :param keys: list of keys or prefixes to iterate over. If None, iterates over all keys.
        """
        if keys is None:
            keys = list(self._store.keys())
        else:
            resolved_keys = []
            for key in keys:
                if key in self._store:
                    resolved_keys.append(key)
                else:
                    resolved_keys.extend(self.iter_keys(key))
            keys = resolved_keys

        num_keys = len(keys)
        if num_keys == 0:
            return

        # Create a dedicated CUDA stream for H2D transfers so they run on a
        # separate stream from the main thread's compute stream. Without this,
        # both threads default to the null stream (stream 0) which serializes
        # all operations and prevents any overlap.
        h2d_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def _fetch_and_record(key):
            event = None
            if h2d_stream is not None:
                with torch.cuda.stream(h2d_stream):
                    data = self._onload_value(self._store[key])
                event = torch.cuda.Event()
                event.record(h2d_stream)
            else:
                data = self._onload_value(self._store[key])
            return data, event

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = None
            for idx, key in enumerate(keys):
                if future is not None:
                    current, event = future.result()
                else:
                    current, event = _fetch_and_record(key)
                if idx + 1 < num_keys:
                    future = executor.submit(_fetch_and_record, keys[idx + 1])
                else:
                    future = None
                # Make the main CUDA stream wait for the background H2D copy
                # before any GPU kernel consumes the prefetched tensors
                if event is not None:
                    torch.cuda.current_stream().wait_event(event)
                yield current

    def __iter__(self) -> Generator[Any, None, None]:
        """
        Iterate over all values in the cache
        """
        for key in self._store:
            yield self._onload_value(self._store[key])

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._store.clear()

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._store)

    @classmethod
    def _onload_value(cls, value: Any) -> Any:
        """
        Onload a value's tensors to their original devices.

        Uses pytree to flatten, onload tensor leaves wrapped in IntermediateValue,
        then unflatten back to original structure.

        :param value: pytree with IntermediateValue at tensor leaf positions
        :return: original value with tensors onloaded
        """
        leaves, spec = tree_flatten(value)
        onloaded_leaves = []
        for leaf in leaves:
            if isinstance(leaf, IntermediateValue):
                tensor = leaf.value
                device = leaf.device
                if device is not None:
                    non_blocking = (
                        tensor.is_pinned()
                        and torch.accelerator.is_available()
                        and torch.device(device).type
                        == torch.accelerator.current_accelerator().type
                    )
                    onloaded_leaves.append(tensor.to(device=device, non_blocking=non_blocking))
                else:
                    onloaded_leaves.append(tensor)
            else:
                onloaded_leaves.append(leaf)
        return tree_unflatten(onloaded_leaves, spec)

    @classmethod
    def _offload_value(
        cls,
        value: Any,
        offload_device: torch.device | None,
        onload_device: torch.device | None = None,
    ) -> Any:
        """
        Offload a value's tensors to the offload device.

        Uses pytree to flatten, wrap tensor leaves in IntermediateValue,
        then unflatten back to original structure.

        :param value: value to offload
        :param offload_device: device to offload tensors to
        :param onload_device: device to onload tensors to. If None, use tensor's current device
        :return: pytree with IntermediateValue at tensor leaf positions
        """
        leaves, spec = tree_flatten(value)
        offloaded_leaves = []
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                with OverrideEqMode():
                    # check for cache hit between shared tensors
                    if leaf in cls.offload_values:
                        offloaded_tensor = cls.offload_values[leaf]
                    else:
                        # move to offload if no hit
                        offloaded_tensor = leaf.to(device=offload_device)
                        if offloaded_tensor is not leaf: # avoid circular ref
                            # pin CPU tensors so onload can use non_blocking DMA
                            if (
                                torch.device(offload_device).type == "cpu"
                                and torch.accelerator.is_available()
                                and not offloaded_tensor.is_pinned()
                            ):
                                offloaded_tensor = offloaded_tensor.pin_memory()
                            cls.offload_values[leaf] = offloaded_tensor

                offloaded_leaves.append(
                    IntermediateValue(
                        value=offloaded_tensor,
                        device=(onload_device if onload_device else leaf.device),
                    )
                )
            elif isinstance(leaf, IntermediateValue):
                warnings.warn(
                    f"Unexpected IntermediateValue in input to _offload_value; "
                    f"passing through unchanged"
                )
                offloaded_leaves.append(leaf)
            else:
                if not isinstance(
                    leaf,
                    (int, str, float, bool, torch.dtype, torch.device, type(None)),
                ):
                    warnings.warn(
                        f"Leaf type {type(leaf)} may not be handled correctly; "
                        f"storing as-is"
                    )
                offloaded_leaves.append(leaf)
        return tree_unflatten(offloaded_leaves, spec)


class OverrideEqMode(TorchDispatchMode):
    """
    When using a torch.Tensor as a key in a dictionary, the equality
    check must return a single value instead of a torch.Tensor
    of bool values.
    Use this override context for such cases, to swap out the torch.eq
    equality check for a check on id
    >>> a = torch.tensor([1,2,3])
    >>> b = torch.tensor([1,2,3])
    >>> a == b
    tensor([True, True, True])
    >>> with OverrideEqMode():
    ...     a == b
    tensor(True)
    """

    def __torch_dispatch__(self, func, _types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # Check if the operation is equality
        if func is torch.ops.aten.eq.Tensor:
            # Override to use torch.equal
            assert len(args) == 2, "Exactly 2 args must be provided"

            # NOTE: Errors out without cast to torch.tensor
            return torch.tensor(id(args[0]) == id(args[1]))

        # For all other operations, just run them normally
        return func(*args, **kwargs)
