"""
Base class for expert pruning modifiers for MoE models.

``ExpertPruningModifierBase`` implements the calibration lifecycle shared by
expert pruning modifiers (currently REAP and MoNe): it identifies the model's
MoE layers, attaches the subclass's activation capture hooks for the
calibration epoch, hands each layer to the subclass's pruning implementation
when its statistics are complete (``SEQUENTIAL_EPOCH_END``), and on finalize
records the pruning in the model's expert-sparsity config and writes the
per-expert statistics the subclass recorded to ``report_path``.

Subclasses accumulate their saliency statistics as private attributes keyed
by layer name (populated in an ``on_calibration_start`` super call) and
implement:

- ``_expert_hook`` and ``_experts_block_hook``: capture calibration
  activations into those statistics
- ``_prune_moe_layer``: use the statistics to prune a completed layer's
  experts and populate ``_report_values`` for report writing

They may extend ``on_initialize`` (e.g. to reject unsupported routers) via a
super call, and must set ``sparsity_method`` to the
:class:`~compressed_tensors.config.MoESparsityMethod` their pruning records
in the model's expert-sparsity config.
"""

import json
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

import torch
from compressed_tensors.config import ExpertSparsityConfig, MoESparsityMethod
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    is_source_process,
    wait_for_comms,
)
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator
from torch import distributed as dist

from llmcompressor.core import Event, State
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.utils import (
    MoeModelAttrs,
    build_moe_num_experts,
    get_moe_attrs,
)

__all__ = ["ExpertPruningModifierBase"]


class ExpertPruningModifierBase(Modifier):
    """
    Base class for modifiers which prune experts from MoE layers based on
    per-expert saliency statistics accumulated over calibration data.

    The base class owns the calibration lifecycle: it identifies the model's
    MoE layers (``on_initialize``), attaches the subclass's activation capture
    hooks (``on_calibration_start``), finalizes each layer once its statistics
    are complete (``on_sequential_epoch_end``), and records the pruning in the
    model's expert-sparsity config and writes the per-expert values the
    subclass recorded in ``_report_values`` to ``report_path``
    (``on_finalize``).

    Subclasses implement the capture hooks (``_expert_hook``,
    ``_experts_block_hook``) which turn routed activations into saliency
    statistics held in their own private attributes, and ``_prune_moe_layer``
    which uses those statistics to prune a completed layer and populate
    ``_report_values`` for report writing.

    :param sparsity: fraction of experts to remove per layer, in ``[0, 1)``.
        When ``0.0``, no experts are removed; saliency is still computed so
        that a ``report_path`` can be written, but the model weights and
        config are never modified.
    :param ignore: module name patterns to skip during MoE layer detection.
    :param report_path: optional path to a ``.json`` file where the per-expert
        statistics are written after calibration completes. The report is a
        JSON object mapping ``count`` and ``saliency`` to one inner list per
        MoE layer (ordered by layer). Required when ``sparsity`` is ``0.0``,
        since the report is then the modifier's only output.
    """

    requires_calibration_data: bool = True

    sparsity: float = 0.0
    report_path: Optional[str] = Field(default=None)

    # abstract attributes for subclasses
    sparsity_method: ClassVar[MoESparsityMethod]

    # private attributes
    _moe_attrs: MoeModelAttrs
    _n_pruned: int
    _n_retained: int
    _report_values: dict[str, dict[str, list[int | float]]] = PrivateAttr(
        default_factory=lambda: defaultdict(dict)
    )
    _pending_layers: set[str] = PrivateAttr(default_factory=set)

    @model_validator(mode="after")
    def _validate_sparsity(self) -> "ExpertPruningModifierBase":
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}")
        return self

    @model_validator(mode="after")
    def _validate_report_path(self) -> "ExpertPruningModifierBase":
        if self.report_path is not None:
            if not self.report_path.endswith(".json"):
                raise ValueError(
                    f"report_path must end with .json, got {self.report_path}"
                )
            Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def _validate_noop(self) -> "ExpertPruningModifierBase":
        if self.sparsity == 0.0 and self.report_path is None:
            raise ValueError(
                f"sparsity is 0.0 (no experts will be pruned) and no report_path is "
                f"set; {type(self).__name__} would do nothing. Set sparsity > 0 "
                f"to {self._expert_verb} experts, or set report_path to "
                "write a saliency report"
            )
        return self

    # -- lifecycle ------------------------------------------------------------

    def on_initialize(self, state: State, **kwargs) -> bool:
        model = state.model
        self._moe_attrs = get_moe_attrs(model)
        self._n_pruned = int(self._moe_attrs.num_experts * self.sparsity)
        self._n_retained = self._moe_attrs.num_experts - self._n_pruned

        # sparsity is too low
        if self.sparsity > 0.0 and self._n_pruned == 0:
            raise ValueError(
                f"sparsity={self.sparsity} results in 0 "
                f"experts to {self._expert_verb} "
                f"(out of {self._moe_attrs.num_experts}). "
                "No pruning will be performed."
            )
        
        # sparsity is too high
        if self._n_retained < self._moe_attrs.top_k:
            raise ValueError(
                f"{type(self).__name__} sparsity is too aggressive: the router "
                f"selects top_k={self._moe_attrs.top_k} experts per token, "
                f"but only {self._n_retained} experts would remain reachable "
                f"after pruning (num_experts={self._moe_attrs.num_experts}, "
                f"dropping≈{self._n_pruned}). Reduce sparsity target"
            )

        logger.info(
            f"{type(self).__name__} initialized: "
            f"{len(self._moe_attrs.moe_layer_names)} MoE layers, "
            f"{self._moe_attrs.num_experts} experts/layer, will "
            f"{self._expert_verb} {self._n_pruned} "
            f"({self.sparsity:.0%})"
        )

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        model: torch.nn.Module = state.model

        self._pending_layers = set(self._moe_attrs.moe_layer_names)

        for layer_name in self._moe_attrs.moe_layer_names:
            module: torch.nn.Module = model.get_submodule(layer_name)
            experts: torch.nn.Module = getattr(module, self._moe_attrs.experts_attr)

            # One hook per expert to record this expert's activations in the
            # subclass's per-layer capture buffers
            expert_list = [
                expert for expert in experts.children() if isinstance(expert, ExpertMLP)
            ]  # Filter out the activation function submodule
            for idx, expert in enumerate(expert_list):
                self.register_hook(
                    expert, partial(self._expert_hook, layer_name, idx), "forward"
                )

            # Hook for the experts block to capture the router's top-k
            # routing decisions and weights. This hook must be a forward
            # hook, so that it runs after the individual expert hooks have
            # populated the capture buffers
            self.register_hook(
                experts, partial(self._experts_block_hook, layer_name), "forward"
            )

    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        """Finalize any tracked layer whose saliency statistics are complete:
        hand it to ``_prune_moe_layer``, which records its per-expert values
        and prunes it when ``sparsity > 0``, then release it."""

        model = state.model

        # sort by layer name to keep the sequence of collectives identical
        # across ranks; every rank finalizes every tracked layer, contributing
        # its zero-initialized accumulators where it saw no tokens
        pruned_layers = 0
        for layer_name in sorted(self._pending_layers):
            if self._prune_moe_layer(model, layer_name):
                # the layer's statistics are complete and the subclass has
                # released its per-layer state
                self._pending_layers.discard(layer_name)
                pruned_layers += 1

        if self._n_pruned > 0:
            logger.info(
                f"{type(self).__name__}: {self._expert_verb_past} experts in "
                f"{pruned_layers} layers"
            )

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """Record the expert-sparsity config on the model and write the
        saliency report."""

        model = state.model

        if self._n_pruned > 0:
            num_experts_per_layer = build_moe_num_experts(model)
            # Picked up by the compressed save path and written into
            # config.json's quantization_config.moe_sparsity_config.
            model._moe_sparsity_config = ExpertSparsityConfig(
                method=self.sparsity_method,
                num_experts=num_experts_per_layer,
            )
            logger.info(
                f"{type(self).__name__}: recorded moe_sparsity_config "
                f"(method={self.sparsity_method.value}) with per-layer "
                f"retained counts {num_experts_per_layer}"
            )

        if self.report_path is not None and is_source_process():
            report = {
                key: [
                    self._report_values.get(layer_name, {}).get(key, [])
                    for layer_name in self._moe_attrs.moe_layer_names
                ]
                for key in ("count", "saliency")
            }
            with open(self.report_path, "w") as file:
                json.dump(report, file)

        self._pending_layers.clear()

        return True

    # -- subclass interface ---------------------------------------------------

    @abstractmethod
    def _expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Forward hook on one expert: record its output activations for this
        batch into the subclass's per-layer capture buffers."""

    @abstractmethod
    def _experts_block_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Forward hook on the layer's experts module: combine the router's
        top-k decisions and weights (``args[1]`` and ``args[2]``) with the
        captured activations to update the layer's saliency statistics, then
        clear the layer's capture buffers."""

    @abstractmethod
    def _prune_moe_layer(self, model: torch.nn.Module, layer_name: str) -> bool:
        """Finalize one MoE layer: reduce its saliency statistics, record its
        per-expert values into ``self._report_values[layer_name]`` (read by
        this base class to write ``report_path``), and, when ``sparsity > 0``,
        record the retained experts in ``self._retained_experts[layer_name]``
        and prune the layer's low-saliency experts.

        :return: True if the layer's statistics were complete and it was
            finalized (the subclass must then have released its per-layer
            state); False if the layer has not collected calibration data
            yet, in which case its statistics are kept for a later epoch."""

    # -- shared helpers -------------------------------------------------------

    def _reduce_stats(
        self, tensors: list[torch.Tensor], count: float, device: torch.device
    ) -> float:
        """Reduce one layer's accumulators to the source rank (rank 0), in
        float32 on the device they were accumulated on, over the default
        process group. The count is all-reduced so that every rank holds the
        count summed across ranks, from which all ranks decide identically
        whether the layer collected any samples.

        The accumulators are reduced in place: ``dist.reduce`` only writes
        its result at the destination, so non-source ranks keep their local
        statistics, while the source rank's accumulators become the global
        sums, which their consumers (report, retained-expert selection) read
        immediately after. A rank whose layer has not seen tokens contributes
        its zero-initialized accumulators, keeping the collectives aligned
        across ranks.
        """
        if not is_distributed():
            return count

        count_tensor = torch.tensor(float(count), dtype=torch.float32, device=device)
        pending = [
            dist.reduce(
                tensor,
                dst=get_source_rank(),
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            for tensor in tensors
        ]
        pending.append(
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM, async_op=True)
        )
        wait_for_comms(pending)
        return float(count_tensor.item())

    def _decide_retained_experts(
        self,
        device: torch.device,
        decide: Callable[[], torch.Tensor],
    ) -> list[int]:
        """Compute the retained-expert ids on the source rank (which holds
        the globally reduced statistics) and broadcast the decision to every
        rank, validating the retained count."""
        expected = self._moe_attrs.num_experts - self._n_pruned
        if is_source_process():
            retained = decide()
        else:
            retained = torch.empty(expected, dtype=torch.int, device=device)

        if is_distributed():
            dist.broadcast(retained, src=get_source_rank())

        assert (
            len(retained) == expected
        ), f"Expected {expected} retained experts, got {len(retained)}"

        return retained.tolist()
