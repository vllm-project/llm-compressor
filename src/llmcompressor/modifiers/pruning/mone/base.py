"""
MoNe (Mixture of Novices) pruning modifier for MoE models.

MoNe mirrors REAP's *collapse* structure (see
:class:`~llmcompressor.modifiers.pruning.reap.REAPPruningModifier`) but ranks
experts with its own "fusion" saliency metric
``S_j = ||std_tokens(f_j)||_2 * E_tokens[g_j * 1(j in top-k)]`` accumulated via
calibration hooks. It selects the lowest-saliency experts per layer and,
instead of pruning them, *collapses* them into constant "novice" vectors -- each
expert's mean output, an unbiased estimator of its contribution -- leaving the
router untouched so the novices keep receiving their original routing weights.

See: https://arxiv.org/abs/2507.00390
"""

import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Optional

import torch
from compressed_tensors import align_module_device
from compressed_tensors.config import ExpertSparsityConfig, MoESparsityMethod
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    is_source_process,
)
from compressed_tensors.offload import get_execution_device
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator
from torch import distributed as dist

from llmcompressor.core import Event, State
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.mone.utils import (
    MoNeSaliencyTracker,
    build_moe_num_experts,
    collapse_moe_layer,
)
from llmcompressor.modifiers.pruning.reap.utils import (
    MoeModelAttrs,
    get_moe_attrs,
)

__all__ = ["MoNePruningModifier"]


class MoNePruningModifier(Modifier):
    """
    Collapses low-saliency experts from MoE layers into constant "novice" vectors
    using the MoNe saliency metric. For each expert ``j`` the saliency is

        ``S_j = ||std_tokens(f_j)||_2 * E_tokens[g_j * 1(j in top-k)]``

    the product of the expert's output fluctuation (the L2 norm of the
    per-dimension standard deviation of expert ``j``'s output ``f_j``) and its
    routing score (the frequency-weighted mean router gate weight ``g_j``,
    averaged over all tokens and zero when ``j`` is not in the top-k). The
    lowest-saliency experts per layer are collapsed: their mean
    output over calibration tokens becomes a constant novice vector, retained
    experts are left-shifted, and the router is kept at its original width so
    novices continue to receive their routing weights. The per-layer retained
    counts are recorded in the checkpoint's
    ``quantization_config.moe_sparsity_config`` with ``method="novice"``.

    MoNe runs during the sequential calibration pipeline: routing and expert
    output statistics are accumulated via hooks on the MoE experts, and the
    structural collapse for a layer is executed when it completes
    (``SEQUENTIAL_EPOCH_END``).

    :param sparsity: fraction of experts to collapse per layer, in ``[0, 1)``.
        When ``0.0``, no experts are collapsed; saliency is still computed so
        that a ``report_path`` can be written, but the model weights and
        config are never modified.
    :param ignore: module name patterns to skip during MoE layer detection.
    :param report_path: optional path to a ``.json`` file where the per-expert
        statistics are written after calibration completes. The report is a JSON
        object mapping ``count`` and ``saliency`` to one inner list per MoE layer
        (ordered by layer). Required when ``sparsity`` is ``0.0``, since the
        report is then the modifier's only output.

    Example recipe::

        MoNePruningModifier:
          sparsity: 0.25
    """

    requires_calibration_data: bool = True

    sparsity: float = 0.0
    ignore: list[str] = Field(default_factory=list)
    report_path: Optional[str] = Field(default=None)

    _moe_attrs: MoeModelAttrs | None = PrivateAttr(default=None)
    _saliency_trackers: dict[str, MoNeSaliencyTracker] = PrivateAttr(
        default_factory=dict
    )
    _n_experts_to_drop: int = PrivateAttr(default=0)
    _output_buffers: dict[str, dict[int, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )
    _retained_experts: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _report: dict = PrivateAttr(default_factory=lambda: defaultdict(dict))

    @model_validator(mode="after")
    def _validate_sparsity(self) -> "MoNePruningModifier":
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}")
        return self

    @model_validator(mode="after")
    def _validate_report_path(self) -> "MoNePruningModifier":
        if self.report_path is not None:
            if not self.report_path.endswith(".json"):
                raise ValueError(
                    f"report_path must end with .json, got {self.report_path}"
                )
            Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def _validate_noop(self) -> "MoNePruningModifier":
        if self.sparsity == 0.0 and self.report_path is None:
            raise ValueError(
                "sparsity is 0.0 (no collapsing) and no report_path is set; MoNe "
                "would do nothing. Set sparsity > 0 to collapse experts, or set "
                "report_path to write a saliency report"
            )
        return self

    def on_initialize(self, state: State, **kwargs) -> bool:
        model = state.model
        self._moe_attrs = get_moe_attrs(model, self.ignore)
        self._n_experts_to_drop = int(self._moe_attrs.num_experts * self.sparsity)

        if self._n_experts_to_drop == 0:
            if self.sparsity > 0.0:
                raise ValueError(
                    f"sparsity={self.sparsity} results in 0 "
                    f"experts to collapse (out of {self._moe_attrs.num_experts}). "
                    "No pruning will be performed."
                )
            # sparsity == 0.0: report-only mode (report_path is guaranteed to be
            # set by validation); saliency is still collected, but there is no
            # collapsing to set up
            logger.info(
                f"MoNe initialized: {len(self._moe_attrs.moe_layer_names)} MoE "
                f"layers, {self._moe_attrs.num_experts} experts/layer, no "
                "experts will be collapsed (sparsity=0.0)"
            )
            return True

        # n_group == 1 is a degenerate "group-limited" router: a single group holds
        # every expert, so routing is equivalent to ungrouped top-k and MoNe's uniform
        # collapse is correct. Only genuine group-limited routers (n_group > 1), which
        # would need group-balanced collapse, are unsupported.
        if self._moe_attrs.n_group is not None and self._moe_attrs.n_group > 1:
            raise NotImplementedError(
                "MoNe does not support group-limited routers "
                f"(n_group={self._moe_attrs.n_group}). Group-balanced expert "
                "collapse is not implemented."
            )

        logger.info(
            f"MoNe initialized: {len(self._moe_attrs.moe_layer_names)} MoE layers, "
            f"{self._moe_attrs.num_experts} experts/layer, will collapse "
            f"{self._n_experts_to_drop} ({self.sparsity:.0%}) into novices"
        )

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        model = state.model

        if get_calibrate_all_experts_flag():
            logger.info(
                "MoNe: moe_calibrate_all_experts is enabled, which is not necessary"
                " for MoNe. You can disable it explicitly by setting "
                "moe_calibrate_all_experts=False in your oneshot() call."
            )

        for layer_name in self._moe_attrs.moe_layer_names:
            module = model.get_submodule(layer_name)
            experts = getattr(module, self._moe_attrs.experts_attr)
            router = getattr(module, self._moe_attrs.router_attr)

            # accumulators are zero-initialized up front on the layer's
            # execution device, in float32: they never move devices (they are
            # reduced in place over the default process group) and a layer
            # which has not seen tokens yet still contributes zeros
            with align_module_device(router):
                hidden_dim = router.weight.shape[1]
            self._saliency_trackers[layer_name] = MoNeSaliencyTracker(
                self._moe_attrs.num_experts,
                hidden_dim,
                get_execution_device(experts),
            )
            self._output_buffers[layer_name] = {}

            # One hook per expert to record its per-token output (for the novice
            # mean and the output-fluctuation term), stored in the layer's buffers.
            expert_list = [
                expert for expert in experts.children() if isinstance(expert, ExpertMLP)
            ]
            for idx, expert in enumerate(expert_list):
                self.register_hook(
                    expert, partial(self._expert_hook, layer_name, idx), "forward"
                )

            # Hook for the experts block to capture the router's top-k decisions and
            # weights and fold them into the routing / novice accumulators. It must
            # run after the per-expert hooks have populated the buffers.
            self.register_hook(
                experts, partial(self._experts_block_hook, layer_name), "forward"
            )

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """Record the MoNe expert-sparsity config on the model for serialization."""

        model = state.model

        if self._n_experts_to_drop > 0:
            num_experts_per_layer = build_moe_num_experts(
                model, self._moe_attrs, self._retained_experts
            )
            # Picked up by the compressed save path and written into
            # config.json's quantization_config.moe_sparsity_config.
            model._moe_sparsity_config = ExpertSparsityConfig(
                method=MoESparsityMethod.NOVICE,
                num_experts=num_experts_per_layer,
            )
            logger.info(
                f"MoNe: recorded moe_sparsity_config (method=novice) with per-layer "
                f"retained counts {num_experts_per_layer}"
            )

        if self.report_path is not None and is_source_process():
            report = {
                key: [
                    self._report[layer_name].get(key, [])
                    for layer_name in self._moe_attrs.moe_layer_names
                ]
                for key in ("count", "saliency")
            }
            with open(self.report_path, "w") as file:
                json.dump(report, file)

        self._saliency_trackers.clear()
        self._output_buffers.clear()

        return True

    # -- decision finalization ----------------------------------------------

    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        """Record saliency for any tracked layer whose saliency is complete,
        collapse it when ``sparsity > 0``, then release its activation
        buffers."""

        model = state.model
        expected = self._moe_attrs.num_experts - self._n_experts_to_drop
        n_dropped = self._n_experts_to_drop

        # sort by layer name to keep the sequence of collectives identical
        # across ranks; every rank reduces every tracked layer, contributing
        # its zero-initialized accumulators where it saw no tokens
        trackers = [
            (layer_name, self._saliency_trackers[layer_name])
            for layer_name in sorted(self._saliency_trackers)
        ]

        for layer_name, tracker in trackers:
            tracker.reduce_stats()

            # every rank holds the token count summed across ranks after the
            # reduction, so all decide identically whether the layer
            # collected any samples; layers with no samples anywhere keep
            # their trackers in case data arrives in a later epoch
            if tracker.total_count == 0:
                continue

            # record per-expert statistics for the report (source rank holds the
            # globally reduced statistics)
            if is_source_process():
                self._report[layer_name]["count"] = tracker.output_count.tolist()
                self._report[layer_name]["saliency"] = tracker.saliency.tolist()

            if self._n_experts_to_drop > 0:
                if is_source_process():
                    retained = tracker.compute_retained_experts(self._n_experts_to_drop)
                else:
                    retained = torch.empty(
                        expected, dtype=torch.int, device=tracker.device
                    )

                if is_distributed():
                    dist.broadcast(retained, src=get_source_rank())

                assert (
                    len(retained) == expected
                ), f"Expected {expected} retained experts, got {len(retained)}"

                retained = retained.tolist()
                retained_set = set(retained)
                num_experts = self._moe_attrs.num_experts
                dropped = [i for i in range(num_experts) if i not in retained_set]

                if is_source_process():
                    novices = tracker.compute_novices(dropped)
                else:
                    novices = torch.empty(
                        n_dropped,
                        tracker.hidden_dim,
                        dtype=torch.float32,
                        device=tracker.device,
                    )

                if is_distributed():
                    dist.broadcast(novices, src=get_source_rank())

                self._retained_experts[layer_name] = retained
                collapse_moe_layer(
                    model, layer_name, retained, novices, self._moe_attrs
                )

            # free this layer's accumulators / buffers now
            del self._saliency_trackers[layer_name]
            self._output_buffers.pop(layer_name, None)

    # -- calibration hooks ---------------------------------------------------

    def _expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Record expert ``f_j`` full outputs (for the novice mean and the
        output-fluctuation term) for every token this batch."""
        if layer_name not in self._output_buffers:
            # Layer has already been collapsed
            return
        if isinstance(output, tuple):
            output = output[0]
        self._output_buffers[layer_name][expert_idx] = output.detach()

    def _experts_block_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Combine router decisions with expert outputs to update the routing and
        novice accumulators."""
        top_k_indices = args[1]
        top_k_weights = args[2]

        tracker = self._saliency_trackers.get(layer_name)
        if tracker is None:
            return

        output_buffer = self._output_buffers[layer_name]

        with torch.no_grad():
            tracker.update_routing(top_k_indices, top_k_weights)
            tracker.update_outputs(top_k_indices, output_buffer)

        self._output_buffers[layer_name] = {}
