from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Generator, Generic, TypeVar
from weakref import WeakKeyDictionary

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tqdm import tqdm


TKey = TypeVar("TKey", bound=Any)


@dataclass
class IntermediateValue:
    """
    Dataclass which recursively defines offloaded values and which device to onload to

    :param value: either an offloaded Tensor, an primative value, or a recursable value
    :param device: if the value is a Tensor, then the device to onload the tensor to,
        otherwise None
    """

    value: torch.Tensor | "IntermediateValue" | Any
    device: torch.device | None


class IntermediatesCache(Generic[TKey]):
    """
    Cache which stores intermediate values (activations) produced by batched, sequential
    execution of models. Values are offloaded to the `offload_device` when stored in
    the cache and onloaded to their original device when fetched from the cache. If
    `offload_device` is None, values will not be offloaded at all.

    Currently supports nested offloading of dataclass instances and tuples

    Construct using `from_dataloader` class method or directly with offload_device
    """

    _store: dict[TKey, IntermediateValue]
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
        Initialize a cache with data from the provided dataloader

        This method iterates through all batches in the dataloader and offloads
        them to the specified device. For faster cache preparation, consider:
        - Increasing batch_size to reduce the number of iterations
        - Using num_workers > 0 in the DataLoader for parallel loading (e.g. the
          calibration DataLoader from format_calibration_data uses
          dataloader_num_workers; when > 0, pin_memory and prefetch_factor are
          also set where applicable, which speeds both cache build and calibration)
        - Ensuring data preprocessing is done before creating the dataloader

        :param dataloader: dataloader which generates values to be cached
        :param model_device: device which values will be onloaded to when fetched
        :param offload_device: device to offload values to
        """
        cache = cls(offload_device)
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Preparing cache")):
            for key, value in batch.items():
                cache._store[(batch_idx, key)] = cls._offload_value(
                    value, offload_device, model_device
                )
        return cache

    def fetch(self, key: TKey) -> Any:
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

    def iter_prefetch(
        self, keys: list[TKey] | None = None
    ) -> Generator[torch.Tensor, None, None]:
        """
        Iterate over keys with values prefetched in a background thread.
        Overlaps onload from offload_device with consumption of the current value,
        which can reduce wall-clock time when offloading to CPU.

        When CUDA is available, uses non_blocking transfers (requires pinned CPU
        tensors, set up by _offload_value) and synchronises via CUDA events so the
        main stream waits for each H2D copy before running GPU kernels on the data.

        :param keys: list of keys to iterate over. If None, iterates over all keys.
        """
        if keys is None:
            keys = list(self._store.keys())

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

    def iter_prefetch_grouped(
        self,
        keys: list[TKey],
        group_by: Callable[[TKey], Any],
    ) -> Generator[dict[TKey, Any], None, None]:
        """
        Prefetch values for given keys, grouped by group_by function.

        Keys are grouped in the order they appear in the input list. Each group
        is yielded as a dict mapping TKey to value. The next group is prefetched
        in a background thread while consuming current.

        Uses CUDA stream optimization when available for overlapping H2D transfers.

        :param keys: list of keys to fetch (caller should filter for existence).
            Keys should be ordered such that all keys in a group appear consecutively.
        :param group_by: function that returns the group identifier for a key
        :yield: dict mapping original TKey to onloaded value, one per group
        """
        if not keys:
            return

        grouped_keys = []
        current_group = []
        current_group_id = None

        for key in keys:
            group_id = group_by(key)
            if group_id != current_group_id:
                if current_group:
                    grouped_keys.append(current_group)
                current_group = [key]
                current_group_id = group_id
            else:
                current_group.append(key)

        if current_group:
            grouped_keys.append(current_group)

        num_groups = len(grouped_keys)

        h2d_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def _fetch_group(key_list: list[TKey]) -> tuple[dict[TKey, Any], Any]:
            event = None
            group_dict = {}
            for key in key_list:
                if h2d_stream is not None:
                    with torch.cuda.stream(h2d_stream):
                        group_dict[key] = self._onload_value(self._store[key])
                    event = torch.cuda.Event()
                    event.record(h2d_stream)
                else:
                    group_dict[key] = self._onload_value(self._store[key])
            return group_dict, event

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = None
            for idx in range(num_groups):
                if future is not None:
                    group_dict, event = future.result()
                else:
                    group_dict, event = _fetch_group(grouped_keys[idx])

                if event is not None:
                    torch.cuda.current_stream().wait_event(event)

                yield group_dict

                if idx + 1 < num_groups:
                    future = executor.submit(_fetch_group, grouped_keys[idx + 1])
                else:
                    future = None

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        """
        Iterate over all tensors in the cache

        :return: generator yielding onloaded tensors
        """
        for key in self._store:
            yield self._onload_value(self._store[key])

    def size(self) -> dict[torch.device, int]:
        """
        Returns the memory used by cached values, keyed by device, in bytes

        :return: dictionary mapping torch device to number of bytes in cache
        """
        sizes = defaultdict(lambda: 0)
        memo = set()

        def _size_helper(intermediate: IntermediateValue) -> int:
            value = intermediate.value

            match value:
                case torch.Tensor():
                    if value not in memo:
                        sizes[value.device] += value.nbytes
                    memo.add(value)
                case list() | tuple():
                    for v in value:
                        _size_helper(v)
                case dict():
                    for v in value.values():
                        _size_helper(v)
                case _ if is_dataclass(value):
                    for field in fields(value):
                        _size_helper(getattr(value, field.name))
                case _:
                    # this handles primitive values that don't match any other cases
                    sizes[torch.device("cpu")] += sys.getsizeof(value, 0)

        for value in self._store.values():
            _size_helper(value)

        return dict(sizes)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._store.clear()

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._store)

    @classmethod
    def _onload_value(cls, intermediate: IntermediateValue) -> Any:
        """
        Onload a value's tensors to the onload device

        :param intermediate: intermediates value representation to onload
        :return: original value with tensors onloaded to the onload device
        """
        value = intermediate.value
        device = intermediate.device

        match value:
            case torch.Tensor():
                # use non_blocking when source is pinned and target is an accelerator
                # so the H2D DMA can overlap with compute on a separate stream
                non_blocking = (
                    value.is_pinned()
                    and device is not None
                    and torch.accelerator.is_available()
                    and torch.device(device).type
                    == torch.accelerator.current_accelerator().type
                )
                return value.to(device=device, non_blocking=non_blocking)
            case list():
                return [cls._onload_value(v) for v in value]
            case tuple():
                return tuple(cls._onload_value(v) for v in value)
            case dict():
                return {k: cls._onload_value(v) for k, v in value.items()}
            case _ if is_dataclass(value):
                for field in fields(value):
                    v = getattr(value, field.name)
                    setattr(value, field.name, cls._onload_value(v))
                return value
            case _:
                # handles primitive values that should be returned as is.
                # without this, a MatchError would be raised for unhandled types.
                return value

    @classmethod
    def _offload_value(
        cls,
        value: Any,
        offload_device: torch.device | None,
        onload_device: torch.device | None = None,
    ) -> IntermediateValue:
        """
        Offload a value's tensors to the offload device

        :param value: value to offload
        :param offload_device: device to offload `torch.Tensor` values to
        :param onload_device: device used when onloading `torch.Tensor` values.
            If None is provided, use the tensor's current device
        :return: Instance of IntermediateValue representing the offloaded value
        """
        kwargs = {"offload_device": offload_device, "onload_device": onload_device}
        match value:
            case torch.Tensor():
                with OverrideEqMode():
                    # check for cache hit between shared tensors
                    if value in cls.offload_values:
                        offloaded = cls.offload_values[value]
                    else:
                        # move to offload if no hit
                        offloaded = value.to(device=offload_device)
                        if offloaded is not value:  # avoid circular ref
                            # pin CPU tensors so onload can use non_blocking DMA
                            if (
                                torch.device(offload_device).type == "cpu"
                                and torch.accelerator.is_available()
                                and not offloaded.is_pinned()
                            ):
                                offloaded = offloaded.pin_memory()
                            cls.offload_values[value] = offloaded

                return IntermediateValue(
                    value=offloaded,
                    device=(onload_device if onload_device else value.device),
                )
            case list():
                return IntermediateValue(
                    value=[cls._offload_value(v, **kwargs) for v in value],
                    device=None,
                )
            case tuple():
                return IntermediateValue(
                    value=tuple(cls._offload_value(v, **kwargs) for v in value),
                    device=None,
                )
            case dict():
                return IntermediateValue(
                    value={
                        k: cls._offload_value(v, **kwargs) for k, v in value.items()
                    },
                    device=None,
                )
            case _ if is_dataclass(value):
                for field in fields(value):
                    v = getattr(value, field.name)
                    setattr(value, field.name, cls._offload_value(v, **kwargs))
                return IntermediateValue(value=value, device=None)
            case _:
                # handles primitive values and provides a warning for unsupported types.
                # without this, values trigger a MatchError exception.
                if not isinstance(
                    value,
                    (int, str, float, bool, torch.dtype, torch.device, type(None)),
                ):
                    warnings.warn(f"Offloading not implemented for type {type(value)}.")
                return IntermediateValue(value=value, device=None)


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
