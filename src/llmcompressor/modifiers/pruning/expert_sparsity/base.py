"""
Base class for expert pruning modifiers for MoE models.

``ExpertPruningModifierBase`` implements the expert pruning lifecycle shared
by expert pruning modifiers (currently REAP and MoNe): it identifies the
model's MoE layers and validates the requested sparsity against them
(``on_initialize``), attaches the activation capture hooks and initializes the
subclass's per-layer statistics (``on_calibration_start``), and finalizes each
layer once its statistics are complete (``on_sequential_epoch_end``): it
reduces the statistics across ranks, records the report values, decides which
experts to prune (dropping the ``sparsity`` fraction with the lowest saliency)
and applies that decision to the layer's structure. On ``on_finalize`` the
pruning is recorded in the model's expert-sparsity config and the per-expert
statistics are written to ``report_path``.

A new expert pruning algorithm only has to define its saliency metric, by
implementing ``_init_moe_stats``, ``_update_stats`` and ``_saliency``, and
may optionally override ``_capture_expert_output`` (how each expert's output
is captured) and ``_apply_retained_experts`` (how a retained-experts decision
is applied to a layer's structure). Everything else -- MoE detection,
distributed reduction of the statistics, the pruning decision (including
group-limited routers and broadcasting the decision across ranks), report
writing and expert-sparsity config recording -- is inherited.
"""

import json
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Optional

import torch
from compressed_tensors.config import ExpertSparsityConfig, MoESparsityMethod
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    is_source_process,
    wait_for_comms,
)
from compressed_tensors.offload import get_execution_device
from compressed_tensors.utils.match import is_match
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator
from torch import distributed as dist
from transformers import PreTrainedModel

from llmcompressor.core import Event, State
from llmcompressor.modeling.moe.helpers import MoEConfig
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.expert_sparsity.utils import (
    get_experts_router,
    prune_moe_layer,
)

__all__ = ["ExpertPruningModifierBase"]


class ExpertPruningModifierBase(Modifier):
    """
    Base class for modifiers which prune experts from MoE layers based on
    per-expert saliency statistics accumulated over calibration data.

    The base class owns the expert pruning lifecycle and the decision of
    which experts to prune: the lowest-saliency ``sparsity`` fraction of each
    layer's experts (per router group for group-limited routers), decided
    from the globally reduced statistics on the source rank and broadcast to
    every rank. Subclasses define how the saliency of an expert is computed
    from the activations routed to it.

    :param sparsity: fraction of experts to remove per layer, in ``[0, 1)``.
        When ``0.0``, no experts are removed; saliency is still computed so
        that a ``report_path`` can be written, but the model weights and
        config are never modified.
    :param ignore: module name patterns to skip during MoE layer detection.
    :param report_path: optional path to a ``.json`` file where the per-expert
        statistics are written after calibration completes. The report is a JSON
        object mapping ``count`` and ``saliency`` to one inner list per MoE layer
        (ordered by layer). Required when ``sparsity`` is ``0.0``, since the
        report is then the modifier's only output.
    """

    requires_calibration_data: bool = True
    precision: torch.dtype = torch.float32

    sparsity: float = 0.0
    ignore: list[str] = Field(default_factory=list)
    report_path: Optional[str] = Field(default=None)

    # subclass configuration: the sparsity method recorded in the model's
    # expert-sparsity config, verbs for log and error messages, and whether
    # group-balanced expert selection applies to the subclass
    sparsity_method: ClassVar[MoESparsityMethod]
    supports_group_limited_routers: ClassVar[bool] = True
    stats_to_report: tuple[str] = ("saliency", "counts")

    # model structure, decided in on_initialize
    _moe_config: MoEConfig = PrivateAttr()
    _n_experts_to_drop: int = PrivateAttr(default=0)
    _n_experts_to_drop_per_group: int | None = PrivateAttr(default=None)

    # per-layer stats used to inform pruning decisions
    _moe_stats: dict[str, dict[str, torch.Tensor]] = PrivateAttr(default_factory=dict)

    # pruning outputs
    _retained_experts: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _report: dict[str, dict[str, list[int | float]]] = PrivateAttr(
        default_factory=lambda: defaultdict(defaultdict(list))
    )

    @model_validator(mode="after")
    def _validate(self) -> "ExpertPruningModifierBase":
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}")
        if self.report_path is not None:
            if not self.report_path.endswith(".json"):
                raise ValueError(
                    f"report_path must end with .json, got {self.report_path}"
                )
            Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
        if self.sparsity == 0.0 and self.report_path is None:
            raise ValueError(
                f"sparsity is 0.0 (no experts will be {self._expert_verb_past}) "
                f"and no report_path is set; {type(self).__name__} would do "
                f"nothing. Set sparsity > 0 to {self._expert_verb} experts, or "
                "set report_path to write a saliency report"
            )
        return self

    # -- lifecycle ------------------------------------------------------------

    def on_initialize(self, state: State, **kwargs) -> bool:
        model: PreTrainedModel = state.model
        self._moe_config = MoEConfig.from_config(model.config)
        num_experts = self._moe_config.num_experts
        self._n_experts_to_drop = int(num_experts * self.sparsity)

        # sparsity is too low
        if self._n_experts_to_drop == 0:
            if self.sparsity > 0.0:
                raise ValueError(
                    f"sparsity={self.sparsity} results in 0 experts to prune "
                    f"(out of {num_experts}). No pruning will be performed."
                )

            # report-only path
            logger.info(
                f"{type(self).__name__} initialized with"
                "`sparsity=0.0`, no experts will be pruned, "
                f"saving report to {self.report_path}"
            )
            return True

        # group-limited routers require an equal number of drops per group
        if self._moe_config.n_group is not None:
            if not self.supports_group_limited_routers and self._moe_config.n_group > 1:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support group-limited "
                    f"routers (n_group={self._moe_config.n_group}). "
                    f"Group-balanced expert sparsity is not implemented."
                )

            self._n_experts_to_drop_per_group = round(
                self._n_experts_to_drop / self._moe_config.n_group
            )
            if self._n_experts_to_drop_per_group == 0:
                raise ValueError(
                    f"Group-limited router detected: sparsity={self.sparsity} "
                    "results in 0 experts to prune per group "
                    f"(out of {self._moe_config.group_size}). "
                    "No pruning will be performed."
                )

        # fail fast (before calibration) if the requested sparsity would
        # leave the router unable to select top_k experts per token
        if self._n_experts_to_drop_per_group is not None:
            available = self._moe_config.top_k_group * (
                self._moe_config.group_size - self._n_experts_to_drop_per_group
            )
        else:
            available = num_experts - self._n_experts_to_drop
        if self._moe_config.num_experts_per_tok > available:
            raise ValueError(
                f"{type(self).__name__} sparsity is too aggressive: the router selects "
                f"top_k={self._moe_config.num_experts_per_tok} experts per token, "
                f"but only {available} experts would remain reachable "
                f"after pruning (num_experts={num_experts}, "
                f"dropping≈{self._n_experts_to_drop}). Reduce the sparsity target"
            )

        logger.info(
            f"{type(self).__name__} initialized: "
            f"{num_experts} experts/layer, will be pruned "
            f"{self._n_experts_to_drop} ({self.sparsity:.0%})"
        )
        if self._n_experts_to_drop_per_group is not None:
            logger.info(
                f"Group-limited router detected: will prune "
                f"{self._n_experts_to_drop_per_group} experts/group "
                f"(out of {self._moe_config.group_size} in each of "
                f"{self._moe_config.n_group} groups)"
            )
            if (
                self._n_experts_to_drop_per_group * self._moe_config.n_group
                != self._n_experts_to_drop
            ):
                logger.warning(
                    f"{type(self).__name__}: group-limited routing requires an "
                    f"equal drop per group; dropping "
                    f"{self._n_experts_to_drop_per_group * self._moe_config.n_group} "
                    f"experts instead of the requested "
                    f"{self._n_experts_to_drop} (n_group={self._moe_config.n_group})"
                )
                self._n_experts_to_drop = (
                    self._n_experts_to_drop_per_group * self._moe_config.n_group
                )

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        model: torch.nn.Module = state.model
        num_experts = self._moe_config.num_experts
        hidden_dim = self._moe_config.hidden_dim

        # TODO: move to `on_sequential_epoch_start`
        for moe_name, moe in model.named_modules():
            experts, router = get_experts_router(moe_name, moe)
            if experts is None or router is None:
                continue

            device = get_execution_device(experts)
            self._init_moe_stats(moe_name, num_experts, hidden_dim, device)
            assert "counts" in self._moe_stats[moe_name], "moe stats must use counts"

            # add hooks to experts
            for idx, expert in enumerate(
                [
                    expert
                    for expert in experts.children()
                    if isinstance(expert, ExpertMLP)
                ]
            ):
                _expert_hook = partial(self.expert_hook, moe_name, idx)
                self.register_hook(expert, _expert_hook, "forward")

            # add hook to experts
            _experts_block_hook = partial(self.experts_block_hook, moe_name)
            self.register_hook(experts, _experts_block_hook, "forward")

    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        """ """
        model: torch.nn.Module = state.model

        # sort by layer name to keep the sequence of distributed collectives identical
        pruned_layers = 0
        for moe_name in sorted(self._moe_stats):
            moe_module = model.get_submodule(moe_name)

            # compute saliency and retained experts
            self.reduce_moe_stats(moe_name)
            device = self._moe_stats[moe_name]["counts"].device
            retained, saliency = self.compute_retained_experts(moe_name, device)

            # apply pruning
            is_ignored = is_match(moe_name, moe_module, self.ignore)
            if self._n_experts_to_drop > 0 and not is_ignored:
                self._retained_experts[moe_name] = retained
                self.prune_moe_layer(model, moe_name, retained)
                pruned_layers += 1
            else:
                self._retained_experts[moe_name] = None

            # source rank writes report
            if is_source_process():
                assert moe_name not in self._report, f"duplicate report for {moe_name}"
                self._report[moe_name] = {
                    key: self._moe_stats[moe_name][key].tolist()
                    for key in self.stats_to_report
                }
                self._report[moe_name]["saliency"] = saliency.tolist()

            # free all stats
            del self._moe_stats[moe_name]

        if self._n_experts_to_drop > 0:
            logger.info(
                f"{type(self).__name__}: pruned {self._n_experts_to_drop} experts "
                f"from each of {pruned_layers} layers"
            )

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """Record the expert-sparsity config on the model and write the
        saliency report."""

        model = state.model

        if self._n_experts_to_drop > 0:
            # Picked up by the compressed save path and written into
            # config.json's quantization_config.moe_sparsity_config.
            model._moe_sparsity_config = ExpertSparsityConfig(
                method=self.sparsity_method,
                num_experts=[
                    len(self._retained_experts[moe_name])
                    if self._retained_experts[moe_name] is not None
                    else None
                    for moe_name in sorted(self._retained_experts)
                ],
            )

        if self.report_path is not None and is_source_process():
            with open(self.report_path, "w") as file:
                json.dump(self._report, file)

        self._moe_stats.clear()

        return True

    # -- calibration hooks ---------------------------------------------------

    @abstractmethod
    def expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        raise NotImplementedError()

    @abstractmethod
    def experts_block_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        raise NotImplementedError()

    # -- statistics and saliency -----------------------------------------------

    @abstractmethod
    def init_moe_stats(
        self,
        layer_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        raise NotImplementedError()

    def reduce_moe_stats(self, layer_name: str) -> float:
        if not is_distributed():
            return

        src_rank = get_source_rank()
        pending = [
            (
                dist.reduce(tensor, dst=src_rank, op=dist.ReduceOp.SUM, async_op=True)
                if name != "counts"
                else dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
            )
            for name, tensor in self._moe_stats[layer_name].items()
        ]
        wait_for_comms(pending)

    @abstractmethod
    def saliency(self, layer_name: str) -> torch.Tensor:
        raise NotImplementedError()

    # -- saliency and pruning -------------------------------

    def compute_retained_experts(
        self, moe_name: str, device: torch.device
    ) -> tuple[list[int], torch.Tensor]:
        num_experts = self._moe_config.num_experts
        expected = num_experts - self._n_experts_to_drop

        if is_source_process():
            saliency = torch.nan_to_num(self.saliency(moe_name), nan=-float("inf"))
            if self._n_experts_to_drop_per_group is None:
                _, drop_indices = torch.topk(
                    saliency, self._n_experts_to_drop, largest=False
                )
                drop_set = {int(i) for i in drop_indices.tolist()}
                retained = [i for i in range(num_experts) if i not in drop_set]
            else:
                retained = []
                for group in range(self._moe_config.n_group):
                    lo = group * self._moe_config.group_size
                    grp = saliency[lo : lo + self._moe_config.group_size]
                    _, drop_local = torch.topk(
                        grp, self._n_experts_to_drop_per_group, largest=False
                    )
                    drop_set = {lo + int(i) for i in drop_local.tolist()}
                    retained.extend(
                        i
                        for i in range(lo, lo + self._moe_config.group_size)
                        if i not in drop_set
                    )

            retained = torch.tensor(retained, dtype=torch.int, device=device)
        else:
            saliency = torch.empty(expected, dtype=self.precision, device=device)
            retained = torch.empty(expected, dtype=torch.int, device=device)

        if is_distributed():
            dist.broadcast(saliency, src=get_source_rank())
            dist.broadcast(retained, src=get_source_rank())

        assert (
            len(retained) == expected
        ), f"Expected {expected} retained experts, got {len(retained)}"

        return retained.tolist()

    def prune_moe_layer(
        self,
        model: torch.nn.Module,
        layer_name: str,
        retained: list[int],
    ):
        """Apply a retained-experts decision to a layer's structure. The
        default structurally prunes the layer to the retained experts;
        subclasses may override it to apply the decision differently (e.g.
        MoNe collapses the dropped experts into novices instead)."""
        prune_moe_layer(model, layer_name, retained)
