"""
Base class for expert pruning modifiers for MoE models.

Implements calibration lifecycle as well as provides interfaces for
subclasses to implement their own saliency calculations and pruning
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
from compressed_tensors.utils import TorchDtype
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
from llmcompressor.utils.helpers import numeric_sorted_names

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
    precision: TorchDtype = torch.float32

    sparsity: float = 0.0
    ignore: list[str] = Field(default_factory=list)
    report_path: Optional[str] = Field(default=None)

    # subclass configuration: the sparsity method recorded in the model's
    # expert-sparsity config, whether group-limited routers are supported,
    # and which stats to serialize in the report
    sparsity_method: ClassVar[MoESparsityMethod]
    supports_group_limited_routers: ClassVar[bool] = True
    stats_to_report: tuple[str] = ("saliency", "count")

    # model structure, decided in on_initialize
    _moe_config: MoEConfig = PrivateAttr()
    _n_experts_to_drop: int = PrivateAttr(default=0)
    _n_experts_to_drop_per_group: int | None = PrivateAttr(default=None)

    # per-layer stats used to inform pruning decisions
    _moe_stats: dict[str, dict[str, torch.Tensor]] = PrivateAttr(default_factory=dict)

    # pruning outputs
    _retained_experts: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _report: dict[str, dict[str, list[int | float]]] = PrivateAttr(default_factory=dict)

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
                f"sparsity is 0.0 (no experts will be pruned) "
                f"and no report_path is set; {type(self).__name__} would do "
                f"nothing. Set sparsity > 0 to prune experts, or "
                "set report_path to write a saliency report"
            )
        return self

    # -- lifecycle ------------------------------------------------------------

    def on_initialize(self, state: State, **kwargs) -> bool:
        model: PreTrainedModel = state.model
        # composite (e.g. VL) model configs keep the MoE fields on their text
        # config, so unwrap it before reading them
        config = model.config
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            config = get_text_config()
        self._moe_config = MoEConfig.from_config(config)
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
            self.init_moe_stats(moe_name, num_experts, hidden_dim, device)
            assert "count" in self._moe_stats[moe_name], "moe stats must use counts"

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
        """Finalize each layer whose statistics are complete: reduce them
        across ranks, record the report values, and prune its lowest-saliency
        experts (when ``sparsity > 0``). Layers which have not seen any
        calibration tokens anywhere keep their statistics, in case data
        arrives in a later epoch."""
        model: torch.nn.Module = state.model

        # sort by layer name to keep the sequence of distributed collectives identical
        pruned_layers = 0
        for moe_name in numeric_sorted_names(self._moe_stats):
            moe_module = model.get_submodule(moe_name)

            # reduce stats and process any MoEs that were calibrated
            self.reduce_moe_stats(moe_name)
            if self._moe_stats[moe_name]["count"].count_nonzero().item() <= 0:
                continue

            # compute saliency and retained experts
            device = self._moe_stats[moe_name]["count"].device
            retained, saliency = self.compute_retained_experts(moe_name, device)

            # apply pruning
            is_ignored = is_match(moe_name, moe_module, self.ignore)
            if self._n_experts_to_drop > 0 and not is_ignored:
                self._retained_experts[moe_name] = retained
                self.prune_moe_layer(moe_name, moe_module, retained)
                pruned_layers += 1
            else:
                self._retained_experts[moe_name] = None

            # source rank writes report
            if is_source_process():
                assert moe_name not in self._report, f"duplicate report for {moe_name}"
                self._moe_stats[moe_name]["saliency"] = saliency
                self._report[moe_name] = {
                    key: self._moe_stats[moe_name][key].tolist()
                    for key in self.stats_to_report
                }

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
        model = state.model

        # Warn and nullify any moe layers which were initialized but never calibrated
        if len(self._moe_stats) > 0:
            logger.warning(
                "Found MoE layers which were initialized but "
                f"never calibrated: {self._moe_stats.keys()}"
            )
            for moe_name in self._moe_stats:
                self._retained_experts[moe_name] = None
                self._report[moe_name] = {key: None for key in self.stats_to_report}

        # Picked up by the compressed save path and written into
        # config.json's quantization_config.moe_sparsity_config.
        if self._n_experts_to_drop > 0:
            model._moe_sparsity_config = ExpertSparsityConfig(
                method=self.sparsity_method,
                num_experts=[
                    len(self._retained_experts[moe_name])
                    if self._retained_experts[moe_name] is not None
                    else None
                    for moe_name in numeric_sorted_names(self._retained_experts)
                ],
            )

        # Report needs to be reconstructed, using the sorted moe names
        # as a source of truth for which layer index is being sparsified
        if self.report_path is not None and is_source_process():
            if self._report.keys() != self._retained_experts.keys():
                extra_diff = self._report.keys() - self._retained_experts.keys()
                missing_diff = self._retained_experts.keys() - self._report.keys()
                logger.warning(
                    "Mismatched keys at save time: \n"
                    f"Extra  : {extra_diff}\n"
                    f"Missing: {missing_diff}\n"
                )

            _report = defaultdict(list)
            for moe_name in numeric_sorted_names(self._report):
                for key in self.stats_to_report:
                    _report[key].append(self._report[moe_name][key])

            with open(self.report_path, "w") as file:
                json.dump(_report, file)

        # cleanup
        self._moe_stats.clear()
        self._retained_experts.clear()
        self._report.clear()

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

    def reduce_moe_stats(self, layer_name: str):
        if not is_distributed():
            return

        src_rank = get_source_rank()
        pending = [
            (
                dist.reduce(tensor, dst=src_rank, op=dist.ReduceOp.SUM, async_op=True)
                if name != "count"
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
            # saliency is broadcast from the source rank at its full
            # [num_experts] size; only retained is sized to `expected`
            saliency = torch.empty(num_experts, dtype=self.precision, device=device)
            retained = torch.empty(expected, dtype=torch.int, device=device)

        if is_distributed():
            dist.broadcast(saliency, src=get_source_rank())
            dist.broadcast(retained, src=get_source_rank())

        assert (
            len(retained) == expected
        ), f"Expected {expected} retained experts, got {len(retained)}"

        return retained.tolist(), saliency

    def prune_moe_layer(
        self,
        moe_name: str,
        moe_module: torch.nn.Module,
        retained: list[int],
    ):
        prune_moe_layer(moe_name, moe_module, retained)
