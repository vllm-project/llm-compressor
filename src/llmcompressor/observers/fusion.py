from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
from weakref import ref

import torch
from torch.nn import Module

if TYPE_CHECKING:
    from llmcompressor.observers.base import Observer

__all__ = ["FusionHandler"]

_msg = "Fused module has been garbage collected before its weight was observed"


class FusionHandler:
    """
    Manages fusion bookkeeping for a single Observer.

    Every Observer gets a FusionHandler at init time. When observers
    are fused (e.g. Q/K/V sharing a global_scale), FusionHandler.fuse()
    links their handlers together and stores weak refs to the modules
    whose weights may need lazy observation.
    """

    def __init__(self, observer: "Observer"):
        self._observer: "Observer" = observer
        self._module_ref: ref[Module] | None = None
        self._group: list[FusionHandler] = []
        self._deletion_called: bool = False

    @property
    def is_fused(self) -> bool:
        return len(self._group) > 0

    @property
    def module(self) -> Module | None:
        if self._module_ref is None:
            return None
        return self._module_ref()

    @classmethod
    def fuse(cls, observers_and_modules: Iterable[tuple["Observer", Module]]) -> None:
        """
        Link all observers in the iterable with each other for shared global_scale.
        Sets module weak-references and makes handlers aware of each other.

        :param observers_and_modules: iterable of (observer, module) tuples
        """
        pairs = list(observers_and_modules)
        handlers = []
        for obs, mod in pairs:
            obs.fusion_handler._module_ref = ref(mod)
            handlers.append(obs.fusion_handler)

        for handler in handlers:
            handler._group = [h for h in handlers if h is not handler]

    def get_fused_statistics(self) -> list[dict[str, torch.Tensor]]:
        """
        Returns statistics for every observer in the fusion group
        (including this handler's own observer).

        If a fused observer lacks statistics, runs observation on its
        module's weight via the weak ref stored during fuse().

        :return: list of {"min_vals": Tensor, "max_vals": Tensor} dicts
        """
        all_handlers = [self] + self._group

        for handler in all_handlers:
            obs = handler._observer
            if not obs.has_statistics:
                mod = handler.module
                assert mod is not None, _msg
                obs(mod.weight)

        return [h._observer.get_statistics() for h in all_handlers]

    def maybe_delete_statistics(self) -> None:
        """
        Cooperative statistics deletion. Each handler calls this when it
        is done with fused statistics. When ALL handlers in the group
        have called it, deletes statistics from every observer in the
        group and resets the bookkeeping.

        For unfused observers, deletes statistics immediately.
        """
        if not self.is_fused:
            self._delete_observer_statistics(self._observer)
            return

        self._deletion_called = True

        if all(h._deletion_called for h in self._group):
            self._deletion_called = False
            for h in self._group:
                h._deletion_called = False
                self._delete_observer_statistics(h._observer)
            self._delete_observer_statistics(self._observer)

    @staticmethod
    def _delete_observer_statistics(observer: "Observer") -> None:
        observer.delete_statistics(check_fused=False)
