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
implementing ``_init_layer_stats``, ``_update_stats`` and ``_saliency``, and
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
from compressed_tensors import align_module_device
from compressed_tensors.config import ExpertSparsityConfig, MoESparsityMethod
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    is_source_process,
    wait_for_comms,
)
from compressed_tensors.offload import get_execution_device
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
    from the activations routed to it:

    - ``_init_layer_stats``: create the per-layer statistics
    - ``_update_stats``: accumulate them over each calibration batch
    - ``_saliency``: score each expert from them

    and may optionally override ``_capture_expert_output`` (how each expert's
    output is captured, to bound peak memory) and ``_apply_retained_experts``
    (how a retained-experts decision is applied to a layer's structure; the
    default structurally prunes the layer). They must set ``sparsity_method``
    to the :class:`~compressed_tensors.config.MoESparsityMethod` their
    pruning records in the model's expert-sparsity config, and may set
    ``supports_group_limited_routers`` to ``False`` if group-balanced
    selection is not applicable to them.

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

    sparsity: float = 0.0
    ignore: list[str] = Field(default_factory=list)
    report_path: Optional[str] = Field(default=None)

    # subclass configuration: the sparsity method recorded in the model's
    # expert-sparsity config, verbs for log and error messages, and whether
    # group-balanced expert selection applies to the subclass
    sparsity_method: ClassVar[MoESparsityMethod]
    _expert_verb: ClassVar[str] = "prune"
    _expert_verb_past: ClassVar[str] = "pruned"
    supports_group_limited_routers: ClassVar[bool] = True

    # model structure, decided in on_initialize
    _moe_attrs: MoeModelAttrs = PrivateAttr()
    _n_experts_to_drop: int = PrivateAttr(default=0)
    _n_experts_to_drop_per_group: int | None = PrivateAttr(default=None)

    # per-layer pruning state. ``_layer_stats[layer_name]`` holds the
    # subclass's accumulators and ``_layer_counts[layer_name]`` the total
    # number of samples the layer has seen; ``_layer_devices[layer_name]``
    # and ``_capture[layer_name]`` its execution device and its per-batch
    # captured expert outputs. They never move devices (the statistics are
    # reduced in place over the default process group), and a layer's
    # entries are released as soon as it is finalized
    _layer_stats: dict[str, dict[str, torch.Tensor]] = PrivateAttr(default_factory=dict)
    _layer_counts: dict[str, float] = PrivateAttr(
        default_factory=lambda: defaultdict(float)
    )
    _layer_devices: dict[str, torch.device] = PrivateAttr(default_factory=dict)
    _capture: dict[str, dict[int, torch.Tensor]] = PrivateAttr(default_factory=dict)

    # pruning outputs
    _retained_experts: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _report_values: dict[str, dict[str, list[int | float]]] = PrivateAttr(
        default_factory=lambda: defaultdict(dict)
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
        self._moe_attrs = get_moe_attrs(state.model, self.ignore)
        num_experts = self._moe_attrs.num_experts
        self._n_experts_to_drop = int(num_experts * self.sparsity)

        # sparsity is too low
        if self._n_experts_to_drop == 0:
            if self.sparsity > 0.0:
                raise ValueError(
                    f"sparsity={self.sparsity} results in 0 experts to "
                    f"{self._expert_verb} (out of {num_experts}). "
                    "No pruning will be performed."
                )
            # sparsity == 0.0: report-only mode (report_path is guaranteed to
            # be set by validation); statistics are still collected for the
            # report, but there is no pruning to set up
            logger.info(
                f"{type(self).__name__} initialized: "
                f"{len(self._moe_attrs.moe_layer_names)} MoE layers, "
                f"{num_experts} experts/layer, no experts will be "
                f"{self._expert_verb_past} (sparsity=0.0)"
            )
            return True

        # group-limited routers require an equal number of drops per group
        if self._moe_attrs.n_group is not None:
            if not self.supports_group_limited_routers and self._moe_attrs.n_group > 1:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support group-limited "
                    f"routers (n_group={self._moe_attrs.n_group}). "
                    f"Group-balanced expert {self._expert_verb} is not implemented."
                )

            self._n_experts_to_drop_per_group = round(
                self._n_experts_to_drop / self._moe_attrs.n_group
            )
            if self._n_experts_to_drop_per_group == 0:
                raise ValueError(
                    f"Group-limited router detected: sparsity={self.sparsity} "
                    f"results in 0 experts to {self._expert_verb} per group "
                    f"(out of {self._moe_attrs.group_size}). "
                    "No pruning will be performed."
                )

        # fail fast (before calibration) if the requested sparsity would
        # leave the router unable to select top_k experts per token
        if self._n_experts_to_drop_per_group is not None:
            available = self._moe_attrs.top_k_group * (
                self._moe_attrs.group_size - self._n_experts_to_drop_per_group
            )
        else:
            available = num_experts - self._n_experts_to_drop
        if self._moe_attrs.top_k > available:
            raise ValueError(
                f"{type(self).__name__} sparsity is too aggressive: the router "
                f"selects top_k={self._moe_attrs.top_k} experts per token, "
                f"but only {available} experts would remain reachable "
                f"after pruning (num_experts={num_experts}, "
                f"dropping≈{self._n_experts_to_drop}). Reduce the sparsity target"
            )

        logger.info(
            f"{type(self).__name__} initialized: "
            f"{len(self._moe_attrs.moe_layer_names)} MoE layers, "
            f"{num_experts} experts/layer, will {self._expert_verb} "
            f"{self._n_experts_to_drop} ({self.sparsity:.0%})"
        )
        if self._n_experts_to_drop_per_group is not None:
            logger.info(
                f"Group-limited router detected: will {self._expert_verb} "
                f"{self._n_experts_to_drop_per_group} experts/group "
                f"(out of {self._moe_attrs.group_size} in each of "
                f"{self._moe_attrs.n_group} groups)"
            )
            if (
                self._n_experts_to_drop_per_group * self._moe_attrs.n_group
                != self._n_experts_to_drop
            ):
                logger.warning(
                    f"{type(self).__name__}: group-limited routing requires an "
                    f"equal drop per group; dropping "
                    f"{self._n_experts_to_drop_per_group * self._moe_attrs.n_group} "
                    f"experts instead of the requested "
                    f"{self._n_experts_to_drop} (n_group={self._moe_attrs.n_group})"
                )
                self._n_experts_to_drop = (
                    self._n_experts_to_drop_per_group * self._moe_attrs.n_group
                )

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        model: torch.nn.Module = state.model

        if get_calibrate_all_experts_flag():
            logger.info(
                f"{type(self).__name__}: moe_calibrate_all_experts is enabled, "
                "which is not necessary. You can disable it explicitly by "
                "setting moe_calibrate_all_experts=False in your oneshot() "
                "call. This may result in faster and/or more memory-efficient "
                "calibration, depending on your model and hardware."
            )

        for layer_name in self._moe_attrs.moe_layer_names:
            module: torch.nn.Module = model.get_submodule(layer_name)
            experts: torch.nn.Module = getattr(module, self._moe_attrs.experts_attr)
            router: torch.nn.Module = getattr(module, self._moe_attrs.router_attr)

            # statistics are zero-initialized up front on the layer's
            # execution device, in float32: they never move devices (they are
            # reduced in place over the default process group) and a layer
            # which has not seen tokens yet still contributes zeros
            device = get_execution_device(experts)
            with align_module_device(router):
                hidden_dim = router.weight.shape[1]

            self._layer_counts[layer_name] = 0.0
            self._layer_devices[layer_name] = device
            self._capture[layer_name] = {}
            self._init_layer_stats(
                layer_name, self._moe_attrs.num_experts, hidden_dim, device
            )

            # One hook per expert to record this expert's activations in the
            # layer's capture buffers
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
        """Finalize any tracked layer whose statistics are complete: record
        its per-expert report values, decide and prune (when ``sparsity > 0``)
        its lowest-saliency experts, then release its statistics."""

        model = state.model

        # sort by layer name to keep the sequence of collectives identical
        # across ranks; every rank finalizes every tracked layer, contributing
        # its zero-initialized accumulators where it saw no tokens
        pruned_layers = 0
        for layer_name in sorted(self._layer_stats):
            if self._reduce_layer_stats(layer_name) == 0:
                # every rank holds the count summed across ranks after the
                # reduction, so all decide identically whether the layer
                # collected any samples; layers with no samples anywhere keep
                # their statistics in case data arrives in a later epoch
                continue

            # the source rank holds the globally reduced statistics
            if is_source_process():
                self._report_values[layer_name] = {
                    "count": self._layer_stats[layer_name]["counts"].tolist(),
                    "saliency": self._saliency(layer_name).tolist(),
                }

            if self._n_experts_to_drop > 0:
                retained = self._decide_retained_experts(layer_name)
                self._retained_experts[layer_name] = retained
                self._apply_retained_experts(model, layer_name, retained)
                pruned_layers += 1

            # free this layer's statistics and capture buffers now
            del self._layer_stats[layer_name]
            del self._layer_counts[layer_name]
            del self._layer_devices[layer_name]
            self._capture.pop(layer_name, None)

        if self._n_experts_to_drop > 0:
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

        if self._n_experts_to_drop > 0:
            num_experts_per_layer = build_moe_num_experts(model, self._retained_experts)
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

        self._layer_stats.clear()
        self._layer_counts.clear()
        self._layer_devices.clear()
        self._capture.clear()

        return True

    # -- calibration hooks ---------------------------------------------------

    def _expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Forward hook on one expert: capture its output activations for
        this batch, reduced by ``_capture_expert_output``."""
        captured = self._capture.get(layer_name)
        if captured is None:
            # Layer has already been finalized
            return
        if isinstance(output, tuple):
            output = output[0]
        with torch.no_grad():
            captured[expert_idx] = self._capture_expert_output(output)

    def _experts_block_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Forward hook on the layer's experts module, running after the
        individual expert hooks: combine the router's top-k decisions and
        weights (``args[1]`` and ``args[2]``) with the captured activations
        to update the layer's statistics, then clear its capture buffers."""
        if layer_name not in self._capture:
            # Layer has already been finalized
            return

        topk_indices, topk_weights = args[1:3]
        with torch.no_grad():
            self._update_stats(
                layer_name, topk_indices, topk_weights, self._capture[layer_name]
            )
        self._capture[layer_name] = {}

    # -- pruning decision ------------------------------------------------------

    def _decide_retained_experts(self, layer_name: str) -> list[int]:
        """Decide which experts to prune from a layer: drop the
        ``_n_experts_to_drop`` experts with the lowest saliency (an equal
        number per router group for group-limited routers, so that the router
        can still select ``top_k`` experts per token).

        The decision is computed on the source rank, which holds the globally
        reduced statistics, and broadcast to every rank."""
        num_experts = self._moe_attrs.num_experts
        expected = num_experts - self._n_experts_to_drop

        if is_source_process():
            saliency = torch.nan_to_num(self._saliency(layer_name), nan=-float("inf"))
            if self._n_experts_to_drop_per_group is None:
                _, drop_indices = torch.topk(
                    saliency, self._n_experts_to_drop, largest=False
                )
                drop_set = {int(i) for i in drop_indices.tolist()}
                retained = [i for i in range(num_experts) if i not in drop_set]
            else:
                retained = []
                for group in range(self._moe_attrs.n_group):
                    lo = group * self._moe_attrs.group_size
                    grp = saliency[lo : lo + self._moe_attrs.group_size]
                    _, drop_local = torch.topk(
                        grp, self._n_experts_to_drop_per_group, largest=False
                    )
                    drop_set = {lo + int(i) for i in drop_local.tolist()}
                    retained.extend(
                        i
                        for i in range(lo, lo + self._moe_attrs.group_size)
                        if i not in drop_set
                    )
            # the statistics never move devices, so the source rank builds
            # the decision on the layer's execution device
            retained = torch.tensor(retained, dtype=torch.int, device=saliency.device)
        else:
            retained = torch.empty(
                expected, dtype=torch.int, device=self._layer_devices[layer_name]
            )

        if is_distributed():
            dist.broadcast(retained, src=get_source_rank())

        assert (
            len(retained) == expected
        ), f"Expected {expected} retained experts, got {len(retained)}"

        return retained.tolist()

    # -- statistics reduction --------------------------------------------------

    def _reduce_layer_stats(self, layer_name: str) -> float:
        """Reduce one layer's accumulators to the source rank (rank 0), in
        float32 on the device they were accumulated on, over the default
        process group. The layer's sample count is all-reduced so that every
        rank holds the count summed across ranks, from which all ranks decide
        identically whether the layer collected any samples.

        The accumulators are reduced in place: ``dist.reduce`` only writes
        its result at the destination, so non-source ranks keep their local
        statistics, while the source rank's accumulators become the global
        sums, which their consumers (report, retained-expert selection) read
        immediately after. A rank whose layer has not seen tokens contributes
        its zero-initialized accumulators, keeping the collectives aligned
        across ranks.
        """
        count = self._layer_counts[layer_name]
        if not is_distributed():
            return count

        count_tensor = torch.tensor(
            float(count), dtype=torch.float32, device=self._layer_devices[layer_name]
        )
        pending = [
            dist.reduce(
                tensor,
                dst=get_source_rank(),
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            for tensor in self._layer_stats[layer_name].values()
        ]
        pending.append(
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM, async_op=True)
        )
        wait_for_comms(pending)
        return float(count_tensor.item())

    # -- subclass interface: the saliency metric -------------------------------

    @abstractmethod
    def _init_layer_stats(
        self,
        layer_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        """Initialize ``self._layer_stats[layer_name]`` for one MoE layer.

        The statistics are a dict mapping names to zero-initialized float32
        tensors on ``device`` which accumulate sums over calibration samples;
        the base class reduces every one of them across ranks. The dict must
        contain a ``"counts"`` entry holding the per-expert sample counts,
        which the base class writes to the saliency report."""

    @abstractmethod
    def _update_stats(
        self,
        layer_name: str,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        captured: dict[int, torch.Tensor],
    ):
        """Accumulate one batch into the layer's statistics.

        :param topk_indices: ``[num_tokens, top_k]`` selected expert ids
        :param topk_weights: ``[num_tokens, top_k]`` gate weight per selection
        :param captured: this batch's per-expert captures ``{expert_idx:
            capture}``, where each capture is the output of
            ``_capture_expert_output`` for the tokens routed to that expert
            (in sparse routing mode, experts only see the tokens the router
            sent to them)

        Implementations update ``self._layer_stats[layer_name]`` and add the
        batch's sample count to ``self._layer_counts[layer_name]``, from
        which the base class decides when the layer's statistics are
        complete."""

    @abstractmethod
    def _saliency(self, layer_name: str) -> torch.Tensor:
        """Score each expert of a layer from its statistics: a
        ``[num_experts]`` tensor whose lowest entries mark the experts to
        prune."""

    def _capture_expert_output(self, output: torch.Tensor) -> torch.Tensor:
        """Reduce one expert's ``[num_routed_tokens, hidden_dim]`` output
        activations for this batch, before they are accumulated. The default
        keeps the full output; subclasses whose metric needs less (e.g. REAP
        needs only its L2 norm) may override this to bound peak memory."""
        return output.detach()

    def _apply_retained_experts(
        self,
        model: torch.nn.Module,
        layer_name: str,
        retained: list[int],
    ):
        """Apply a retained-experts decision to a layer's structure. The
        default structurally prunes the layer to the retained experts;
        subclasses may override it to apply the decision differently (e.g.
        MoNe collapses the dropped experts into novices instead)."""
        prune_moe_layer(model, layer_name, retained, self._moe_attrs)
