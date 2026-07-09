"""
REAP (Router-weighted Expert Activation Pruning) modifier for MoE models.

See: https://arxiv.org/abs/2510.13999
"""

from functools import partial
from typing import Any

import torch
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator

from llmcompressor.core import Event, State
from llmcompressor.core.session_functions import active_session
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.reap.utils import (
    MoeModelAttrs,
    REAPSaliencyTracker,
    get_moe_attrs,
    prune_moe_layer,
    update_model_config,
)

__all__ = ["REAPPruningModifier"]


class REAPPruningModifier(Modifier):
    """
    Prunes experts from MoE layers using the REAP saliency metric. For each
    expert ``j`` the saliency is

        ``S_j = mean(g_j * ||f_j||_2)``

    averaged over the tokens routed to expert ``j``, where:

    - ``g_j`` is the router gate weight assigned to expert ``j`` (the coefficient
      that multiplies the expert's output when combining experts), and
    - ``f_j`` is expert ``j``'s output activation for that token, so
      ``||f_j||_2`` is its L2 norm.

    The lowest-saliency experts are removed per layer. REAP runs during the
    sequential calibration pipeline: saliency is accumulated via hooks on the MoE
    experts, the structural pruning for a layer is executed when it
    completes (``SEQUENTIAL_EPOCH_END``). The config is updated to reflect the new
    number of experts in ``on_finalize``.

    :param sparsity: fraction of experts to remove per layer (0, 1).
    :param ignore: module name patterns to skip during MoE layer detection.

    Example recipe::

        REAPPruningModifier:
          sparsity: 0.25
    """

    sparsity: float
    ignore: list[str] = Field(default_factory=list)

    _moe_attrs: MoeModelAttrs | None = PrivateAttr(default=None)
    _saliency_trackers: dict[str, REAPSaliencyTracker] = PrivateAttr(
        default_factory=dict
    )
    _n_experts_to_drop: int = PrivateAttr(default=0)
    _n_experts_to_drop_per_group: int | None = PrivateAttr(default=None)
    _norm_buffers: dict[str, dict[int, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _validate_sparsity(self) -> "REAPPruningModifier":
        if not 0.0 < self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in (0, 1), got {self.sparsity}")
        return self

    def on_initialize(self, state: State, **kwargs) -> bool:
        model = state.model

        self._moe_attrs = get_moe_attrs(model, self.ignore)

        self._n_experts_to_drop = int(self._moe_attrs.num_experts * self.sparsity)

        if self._n_experts_to_drop == 0:
            raise ValueError(
                f"sparsity={self.sparsity} results in 0 "
                f"experts to drop (out of {self._moe_attrs.num_experts}). "
                "No pruning will be performed."
            )

        if self._moe_attrs.n_group is not None:
            self._n_experts_to_drop_per_group = round(
                self._n_experts_to_drop / self._moe_attrs.n_group
            )

            if self._n_experts_to_drop_per_group == 0:
                raise ValueError(
                    f"Group-limited router detected: sparsity={self.sparsity} "
                    f"results in 0 experts to drop per group "
                    f"(out of {self._moe_attrs.group_size}). "
                    "No pruning will be performed."
                )

        # fail fast (before calibration) if the requested sparsity would
        # leave the router unable to select top_k experts per token
        if self._moe_attrs.n_group is not None:
            retained_per_group = (
                self._moe_attrs.group_size - self._n_experts_to_drop_per_group
            )
            available = self._moe_attrs.top_k_group * retained_per_group
        else:
            available = self._moe_attrs.num_experts - self._n_experts_to_drop

        if self._moe_attrs.top_k > available:
            raise ValueError(
                f"REAP sparsity is too aggressive: the router selects "
                f"top_k={self._moe_attrs.top_k} experts per token, "
                f"but only {available} experts would remain reachable "
                f"after pruning (num_experts={self._moe_attrs.num_experts}, "
                f"dropping≈{self._n_experts_to_drop}). Reduce the sparsity"
            )

        logger.info(
            f"REAP initialized: {len(self._moe_attrs.moe_layer_names)} MoE layers, "
            f"{self._moe_attrs.num_experts} experts/layer, will drop "
            f"{self._n_experts_to_drop} ({self.sparsity:.0%})"
        )
        if self._n_experts_to_drop_per_group is not None:
            logger.info(
                f"Group-limited router detected: will drop "
                f"{self._n_experts_to_drop_per_group} experts/group "
                f"(out of {self._moe_attrs.group_size} in each of "
                f"{self._moe_attrs.n_group} groups)"
            )
            if (
                self._n_experts_to_drop_per_group * self._moe_attrs.n_group
                != self._n_experts_to_drop
            ):
                logger.warning(
                    f"REAP: group-limited routing requires an equal drop "
                    f"per group; dropping "
                    f"{self._n_experts_to_drop_per_group * self._moe_attrs.n_group} "
                    f"experts instead of the requested "
                    f"{self._n_experts_to_drop} "
                    f"(n_group={self._moe_attrs.n_group})"
                )
                self._n_experts_to_drop = (
                    self._n_experts_to_drop_per_group * self._moe_attrs.n_group
                )

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        model = state.model

        # Ensure that REAP is the only modifier for this calibration pass
        session = active_session()
        if len(session.lifecycle.recipe.modifiers) > 1:
            raise ValueError(
                "REAPPruningModifier must be the only modifier in the recipe "
                "during calibration. Other modifiers may interfere with "
                "the saliency tracking and pruning. Please only use one modifier "
                "or use the independent pipeline."
            )

        # Info if moe_calibrate_all_experts is enabled, which is not necessary for REAP
        if get_calibrate_all_experts_flag():
            logger.info(
                "REAP: moe_calibrate_all_experts is enabled, which is not necessary"
                " for REAP. You can disable it explicity by setting "
                "moe_calibrate_all_experts=False in your oneshot() call. This may "
                "result in faster and/or more memory-efficient calibration, "
                "depending on your model and hardware."
            )

        for layer_name in self._moe_attrs.moe_layer_names:
            module = model.get_submodule(layer_name)

            self._saliency_trackers[layer_name] = REAPSaliencyTracker(
                self._moe_attrs.num_experts
            )
            self._norm_buffers[layer_name] = {}

            # One hook per expert to record its per-token output norm and
            # store in the layer's norm buffer
            experts = getattr(module, self._moe_attrs.experts_attr)
            expert_list = [
                expert for expert in experts.children() if isinstance(expert, ExpertMLP)
            ]  # Filter out the activation function submodule
            for idx, expert in enumerate(expert_list):
                self.register_hook(
                    expert, partial(self._expert_hook, layer_name, idx), "forward"
                )

            # Hook for the experts block to capture the router's top-k
            # routing decisions and weights. This hook also executes pruning,
            # which requires the expert output norms. Therefore, it must be a
            # forward hook, so that it runs after the individual expert hooks
            # have populated the norm buffer
            self.register_hook(
                experts, partial(self._experts_block_hook, layer_name), "forward"
            )

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """Finalize the model config to reflect the new number of experts."""

        model = state.model

        new_num_experts = self._moe_attrs.num_experts - self._n_experts_to_drop
        update_model_config(model, self._moe_attrs, new_num_experts)

        self._saliency_trackers.clear()
        self._norm_buffers.clear()

        return True

    # -- decision finalization ----------------------------------------------

    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        """Prune any tracked layer whose saliency is
        complete, then release its activation norm buffers."""

        model = state.model

        for layer_name, tracker in list(self._saliency_trackers.items()):
            if tracker.total_count <= 0:
                continue

            retained = tracker.compute_retained_experts(
                self._n_experts_to_drop,
                self._n_experts_to_drop_per_group,
                self._moe_attrs,
            )
            expected = self._moe_attrs.num_experts - self._n_experts_to_drop
            assert len(retained) == expected, (
                f"Expected {expected} retained experts, got {len(retained)}"
            )

            prune_moe_layer(model, layer_name, retained, self._moe_attrs)

            # free this layer's accumulators / buffers now
            del self._saliency_trackers[layer_name]
            self._norm_buffers.pop(layer_name, None)

    # -- calibration hooks ---------------------------------------------------

    def _expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Record expert ``f_j`` output norms for every token this batch."""
        if layer_name not in self._norm_buffers:
            # Layer has already been pruned
            return
        if isinstance(output, tuple):
            output = output[0]
        with torch.no_grad():
            norms = torch.linalg.norm(output.float(), dim=-1).reshape(-1)
        self._norm_buffers[layer_name][expert_idx] = norms

    def _experts_block_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Combine router decisions with expert norms to update saliency."""
        top_k_indices = args[1]
        top_k_weights = args[2]

        tracker = self._saliency_trackers.get(layer_name)
        if tracker is None:
            return

        norm_buffer = self._norm_buffers[layer_name]

        with torch.no_grad():
            tracker.update(top_k_indices, top_k_weights, norm_buffer)

        self._norm_buffers[layer_name] = {}
