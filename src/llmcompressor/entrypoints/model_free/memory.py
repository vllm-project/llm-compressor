import sys
import traceback as tb
import weakref
from functools import partial
from types import TracebackType
from typing import Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

__all__ = ["TensorProfiler"]


class MemoryProfile:
    _timelines: dict[torch.device, list[int]]

    def __init__(self):
        self._timelines: dict[torch.device, list[int]] = dict()

    def add(self, device: torch.device, size: int):
        if device not in self._timelines:
            timeline = [0 for _ in range(max(len(self), 1))]
            self._timelines[device] = timeline

        for dev in self._timelines:
            if dev == device:
                diff = size
            else:
                diff = 0

            self._timelines[dev].append(self._timelines[dev][-1] + diff)

    def subtract(self, device: torch.device, size: int):
        self.add(device, -size)

    @property
    def current(self) -> dict[torch.device, int]:
        return {device: self._timelines[device][-1] for device in self._timelines}

    @property
    def peak(self) -> dict[torch.device, int]:
        return {device: max(self._timelines[device]) for device in self._timelines}

    def __len__(self) -> int:
        return max((len(timeline) for timeline in self._timelines.values()), default=0)


class TensorProfiler(TorchDispatchMode):
    _tracked: set[int]
    _memory: MemoryProfile
    _exception: BaseException | None

    def __init__(self, catch_exception: bool = True):
        self._tracked = set()
        self._memory = MemoryProfile()
        self._catch_exception = catch_exception
        self._exception = None

    # ::::::::::::::::::::::::::::::::::::::::::::::::
    # 📤 Public API — user-facing methods
    # ::::::::::::::::::::::::::::::::::::::::::::::::

    @property
    def memory(self) -> dict[torch.device | str, int]:
        ret = self._memory.current.copy()
        total = sum(ret.values(), start=0)
        ret.update({"total": total})
        return ret

    @property
    def memory_peak(self) -> dict[torch.device | str, int]:
        ret = self._memory.peak.copy()
        all = max(ret.values(), default=0)
        ret.update({"all": all})
        return ret

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    # ::::::::::::::::::::::::::::::::::::::::::::::::
    # ⚙️ Tracking - Dispatch overload and finalizers
    # ::::::::::::::::::::::::::::::::::::::::::::::::

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        ret = func(*args, **(kwargs or {}))

        for obj in tree_leaves(ret):
            if isinstance(obj, torch.Tensor):
                self._track(obj.untyped_storage())

        return ret

    def _track(self, storage: torch.UntypedStorage):
        hash = storage._cdata
        size = storage.nbytes()
        device = storage.device

        # skip if already tracked
        if hash in self._tracked:
            return

        # track
        self._memory.add(device, size)
        self._tracked.add(hash)

        # register finalizer to subtract memory
        finalizer = partial(self._untrack, hash, size, device)
        weakref.finalize(storage, finalizer)  # triggers regardless of gc

    def _untrack(self, hash: int, size: int, device: torch.device):
        # skip if no longer tracking
        if hash not in self._tracked:
            return

        # untrack
        self._memory.subtract(device, size)
        self._tracked.remove(hash)

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        if self._catch_exception and exc_type is not None:
            self._exception = exc_value
            tb.print_exception(exc_type, exc_value, traceback, file=sys.stderr)

        self._tracked = set()
        return super().__exit__(exc_type, exc_value, traceback) or self._catch_exception
