from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from typing import Any, Generator, Sequence
from weakref import WeakKeyDictionary

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tqdm import tqdm

# Top-level type alias: a list of batches, each batch is a dict string -> Any
IntermediateBatches = list[dict[str, Any]]


class IntermediateCache:
    """
    Cache a single tensor and move it between an offload device and the execution
    device on demand.

    The stored ``value`` is always the currently cached representation — i.e. it lives
    on ``offload_device`` when offloading is enabled.
    """

    # map of onload value -> offload value
    # used to avoid excess memory usage when shared tensors are offloaded
    offload_values: WeakKeyDictionary[torch.Tensor, torch.Tensor] = WeakKeyDictionary()

    def __init__(
        self,
        value: torch.Tensor,
        offload_device: torch.device | None = "cpu",
        onload_device: torch.device | None = None,
    ):
        self.offload_device = offload_device
        self.onload_device = (
            onload_device if onload_device is not None else value.device
        )
        self.value = self._offload_value(value)

    def fetch(self) -> torch.Tensor:
        return self._onload_value(self.value)

    def update(self, value: torch.Tensor):
        self.value = self._offload_value(value)

    def onload_(self, device: torch.device | None = None):
        self.value = self._onload_value(self.value, device)

    def offload_(self):
        self.value = self._offload_value(self.value)

    @contextmanager
    def onloaded(self, device: torch.device | None = None):
        self.onload_(device)
        try:
            yield self.value
        finally:
            self.offload_()

    def size(self) -> dict[torch.device, int]:
        sizes = defaultdict(lambda: 0)
        if isinstance(self.value, torch.Tensor):
            sizes[self.value.device] += self.value.nbytes
        else:
            sizes[torch.device("cpu")] += sys.getsizeof(self.value, 0)
        return dict(sizes)

    def _onload_value(
        self,
        value: torch.Tensor,
        device_override: torch.device | None = None,
    ) -> torch.Tensor:
        device = device_override if device_override is not None else self.onload_device
        if device is None:
            return value
        non_blocking = value.is_pinned() and torch.device(device).type == "cuda"
        return value.to(device=device, non_blocking=non_blocking)

    def _offload_value(self, value: torch.Tensor) -> torch.Tensor:
        """
        Offload a tensor to the offload device.

        If offload_device is None or same as onload_device, returns value unchanged.
        """
        offload_device = self.offload_device

        # No-op if offload_device is None or same as onload_device
        if offload_device is None or offload_device == self.onload_device:
            return value

        with OverrideEqMode():
            if value in self.offload_values:
                offloaded = self.offload_values[value]
            else:
                offloaded = value.to(device=offload_device)
                if offloaded is not value:
                    if (
                        offload_device is not None
                        and torch.device(offload_device).type == "cpu"
                        and torch.cuda.is_available()
                        and not offloaded.is_pinned()
                    ):
                        offloaded = offloaded.pin_memory()
                    self.offload_values[value] = offloaded
        return offloaded


def empty_batches(num_batches: int) -> IntermediateBatches:
    return [{} for _ in range(num_batches)]


def recursive_offload_to_cache(
    value: Any,
    offload_device: torch.device | None,
    onload_device: torch.device | None = None,
) -> Any:
    """
    Recursively offload any tensors in a nested structure of lists, tuples and dicts

    :param value: value to offload
    :param offload_device: device to offload `torch.Tensor` values to
    :return: value with all tensors offloaded
    """
    kwargs = {"offload_device": offload_device, "onload_device": onload_device}
    match value:
        case torch.Tensor():
            return IntermediateCache(value, **kwargs)
        case list():
            return [recursive_offload_to_cache(v, **kwargs) for v in value]
        case tuple():
            return tuple(recursive_offload_to_cache(v, **kwargs) for v in value)
        case dict():
            return {
                k: recursive_offload_to_cache(v, **kwargs) for k, v in value.items()
            }
        case _ if is_dataclass(value):
            for field in fields(value):
                v = getattr(value, field.name)
                setattr(value, field.name, recursive_offload_to_cache(v, **kwargs))
            return value
        case _:
            # handles primitive values and provides a warning for unsupported types.
            # without this, values trigger a MatchError exception.
            if not isinstance(
                value,
                (int, str, float, bool, torch.dtype, torch.device, type(None)),
            ):
                warnings.warn(f"Offloading not implemented for type {type(value)}.")
            return value


def build_batches_from_dataloader(
    dataloader: torch.utils.data.DataLoader,
    model_device: torch.device = torch.device("cpu"),
    offload_device: torch.device | None = torch.device("cpu"),
) -> IntermediateBatches:
    """
    Materialize a dataloader into cached batches of ``IntermediateCache`` values.
    """
    batch_intermediates = []
    for batch in tqdm(dataloader, desc="Preparing cache"):
        batch_cache = {
            key: recursive_offload_to_cache(value, offload_device, model_device)
            for key, value in batch.items()
        }
        batch_intermediates.append(batch_cache)
    return batch_intermediates


def recursive_fetch_from_cache(value: Any) -> Any:
    match value:
        case IntermediateCache():
            return value.fetch()
        case list():
            return [recursive_fetch_from_cache(v) for v in value]
        case tuple():
            return tuple(recursive_fetch_from_cache(v) for v in value)
        case dict():
            return {k: recursive_fetch_from_cache(v) for k, v in value.items()}
        case _ if is_dataclass(value):
            for field in fields(value):
                v = getattr(value, field.name)
                setattr(value, field.name, recursive_fetch_from_cache(v))
            return value
        case _:
            return value


def fetch_batch(
    batch: dict[str, Any], input_names: list[str] | None = None
) -> dict[str, Any]:
    return {
        key: recursive_fetch_from_cache(cache)
        for key, cache in batch.items()
        if input_names is None or key in input_names
    }


def update_batch(
    batch: dict[str, Any],
    values: dict[str, Any],
    offload_device: torch.device | None = "cpu",
) -> None:
    batch.update(
        {k: recursive_offload_to_cache(v, offload_device) for k, v in values.items()}
    )


def delete_from_batch(
    batch: dict[str, Any], consumed_names: list[str] | None = None
) -> None:
    if consumed_names is None:
        consumed_names = list(batch.keys())

    for name in consumed_names:
        del batch[name]


def iter_batches(
    batches: IntermediateBatches, input_names: list[str] | None = None
) -> Iterator[dict[str, Any]]:
    for batch in batches:
        yield fetch_batch(batch, input_names)


def maybe_prefetch(batches: Sequence[Any]) -> Iterator[Any]:
    """
    Iterate with optional one-item background prefetch, controlled by
    ``active_session().state.sequential_prefetch``.

    When CUDA is available, records an event on the worker thread's current stream
    so the main thread can wait for any asynchronous H2D copies before consuming
    the prefetched item.
    """

    try:
        from llmcompressor.core import active_session

        use_prefetch = active_session().state.sequential_prefetch
    except Exception:
        use_prefetch = False

    if use_prefetch:
        # Single ThreadPoolExecutor for all caches
        yield from _prefetch_all(batches)
    else:
        # Direct fetch - replace each cache with its fetched value
        for batch in batches:
            yield recursive_fetch_from_cache(batch)


def _prefetch_all(batches: Sequence[Any]) -> Generator[Any, None, None]:
    """Prefetch all caches in a single ThreadPoolExecutor."""

    # Create a dedicated CUDA stream for H2D transfers so they run on a
    # separate stream from the main thread's compute stream. Without this,
    # both threads default to the null stream (stream 0) which serializes
    # all operations and prevents any overlap.
    h2d_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _fetch_and_record(batch):
        event = None
        if h2d_stream is not None:
            with torch.cuda.stream(h2d_stream):
                data = fetch_batch(batch)
            event = torch.cuda.Event()
            event.record(h2d_stream)
        else:
            data = fetch_batch(batch)
        return data, event

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for batch_index, batch in enumerate(batches):
            if future is not None:
                current, event = future.result()
            else:
                current, event = _fetch_and_record(batch)
            if batch_index + 1 < len(batches):
                future = executor.submit(_fetch_and_record, batches[batch_index + 1])
            else:
                future = None
            # Make the main CUDA stream wait for the background H2D copy
            # before any GPU kernel consumes the prefetched tensors
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            yield current


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
