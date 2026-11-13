from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch.nn import Module


class ModuleTransform(ABC):
    """Model-wide transform that owns original/meta module pairs by FQN."""

    def __init__(self):
        self.modules: dict[str, tuple[Module, Module]] = {}
        self.source_modules: dict[str, dict[str, Module]] = {}
        self.offload_kwargs: dict[str, dict] = {}
        self.source_offloaded: set[str] = set()
        self.materialized: set[str] = set()

    @abstractmethod
    def to_meta(self, model: Module, excluded_fqns: set[str]) -> set[str]:
        """Replace matching modules with meta targets and return replaced FQNs."""

    @abstractmethod
    def materialize_module(
        self,
        fqn: str,
        target: Module,
        source: Module,
        device: torch.device,
        offload_kwargs: dict,
    ) -> dict[str, dict]:
        """Populate one initialized target from its original module."""

    def restore_module(
        self,
        model: Module,
        fqn: str,
        source: Module,
        target: Module,
        materialized: bool,
    ) -> None:
        """Restore an unmaterialized source; keep materialized targets by default."""
        if not materialized:
            model.set_submodule(fqn, source)

    def record_module(self, fqn: str, source: Module, target: Module) -> None:
        """Store the module pair and index the original subtree for staging."""
        self.modules[fqn] = (source, target)
        self.source_modules[fqn] = {
            _qualify_fqn(fqn, relative_name): module
            for relative_name, module in source.named_modules()
        }

    def source_modules_for_subgraph(
        self, target_modules: Mapping[str, Module]
    ) -> dict[str, Module]:
        """Return retained original modules used to materialize this subgraph."""
        modules: dict[str, Module] = {}
        for fqn in self.modules:
            if fqn in self.materialized or not subgraph_owns_fqn(
                target_modules, fqn
            ):
                continue
            modules.update(self.source_modules[fqn])
        return modules

    def materialize(
        self,
        target_modules: Mapping[str, Module],
        source_modules: Mapping[str, Module],
        source_offload_kwargs: Mapping[str, dict],
        device: torch.device,
    ) -> dict[str, dict]:
        """Materialize this transform's targets owned by a subgraph."""
        target_offload_kwargs: dict[str, dict] = {}
        for fqn, (source, target) in self.modules.items():
            if fqn in self.materialized or not subgraph_owns_fqn(
                target_modules, fqn
            ):
                continue
            source = source_modules.get(fqn, source)
            offload_kwargs = source_offload_kwargs.get(fqn)
            if offload_kwargs is None:
                offload_kwargs = self.offload_kwargs[fqn]
            relative_kwargs = self.materialize_module(
                fqn, target, source, device, offload_kwargs
            )
            self.materialized.add(fqn)
            for relative_name, kwargs in relative_kwargs.items():
                target_name = _qualify_fqn(fqn, relative_name)
                target_offload_kwargs[target_name] = kwargs
        return target_offload_kwargs

    def restore(self, model: Module, fqns: set[str] | None = None) -> None:
        """Restore or release mapped modules in the selected subgraph."""
        for fqn in list(self.modules):
            if fqns is not None and fqn not in fqns:
                continue
            source, target = self.modules[fqn]
            self.restore_module(model, fqn, source, target, fqn in self.materialized)
            self.forget_module(fqn)

    def forget_module(self, fqn: str) -> None:
        """Drop source and target references after restoration or final use."""
        self.modules.pop(fqn, None)
        self.source_modules.pop(fqn, None)
        self.offload_kwargs.pop(fqn, None)
        self.source_offloaded.discard(fqn)
        self.materialized.discard(fqn)


def _qualify_fqn(root: str, relative_name: str) -> str:
    if not relative_name:
        return root
    return f"{root}.{relative_name}" if root else relative_name


def subgraph_owns_fqn(target_modules: Mapping[str, Module], fqn: str) -> bool:
    return fqn in target_modules or any(
        name.startswith(f"{fqn}.") for name in target_modules
    )


def fqn_overlaps(fqn: str, roots: set[str]) -> bool:
    """Whether a path is equal to, inside, or contains a transformed root."""
    return any(
        fqn == root or fqn.startswith(f"{root}.") or root.startswith(f"{fqn}.")
        for root in roots
    )
