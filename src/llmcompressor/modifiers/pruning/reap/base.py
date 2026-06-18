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
from llmcompressor.modeling.moe import get_calibrate_all_experts_flag
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.reap.utils import (
    MoEModelAttrs,
    REAPSaliencyTracker,
    assert_homogeneous_moe,
    assert_routing_feasible,
    compute_retained_experts,
    detect_moe_attrs,
    extract_routing,
    find_moe_layers,
    get_num_experts,
    get_router_num_groups,
    get_router_topk_group,
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
    calibration wrappers (which route every token through every expert), the
    drop decision for a layer is finalized as soon as its calibration subgraph
    completes (``SEQUENTIAL_EPOCH_END``) so its activation buffers can be freed,
    and the structural pruning is applied at ``on_finalize`` (after the
    calibration context has exited and a layer's second error-propagation pass,
    if any, has run).

    :param sparsity: fraction of experts to remove per layer (0, 1). For
        group-limited routers (DeepSeek-V3, GLM4, GLM-MoE-DSA) the count is
        rounded to an equal number per expert group.
    :param ignore: module name patterns to skip during MoE layer detection.

    Example recipe::

        REAPPruningModifier:
          sparsity: 0.5
    """

    sparsity: float
    ignore: list[str] = Field(default_factory=list)

    _attrs: MoEModelAttrs | None = PrivateAttr(default=None)
    _moe_layer_names: list[str] = PrivateAttr(default_factory=list)
    _saliency_trackers: dict[str, REAPSaliencyTracker] = PrivateAttr(
        default_factory=dict
    )
    _top_k: int = PrivateAttr(default=2)
    _num_experts: int = PrivateAttr(default=0)
    _n_group: int = PrivateAttr(default=1)
    _n_experts_to_drop: int = PrivateAttr(default=0)
    _prune_decisions: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _norm_buffers: dict[str, dict[int, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )
    _routing_cache: dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sparsity(self) -> "REAPPruningModifier":
        if not 0.0 < self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in (0, 1), got {self.sparsity}")
        return self

    def on_initialize(self, state: State, **kwargs) -> bool:
        model = state.model

        self._attrs = detect_moe_attrs(model)
        if self._attrs is None:
            raise ValueError(
                "Could not detect a supported MoE architecture. REAP requires an "
                "MoE model with a calibration wrapper that routes all tokens to "
                "all experts."
            )

        moe_layers = find_moe_layers(model, self._attrs, self.ignore)
        if not moe_layers:
            raise ValueError("No MoE layers found in model.")
        self._moe_layer_names = list(moe_layers.keys())

        config = model.config
        text_config = getattr(config, "text_config", config)
        self._top_k = getattr(text_config, "num_experts_per_tok", 2)

        # fast fail if all layers do not have same num_experts, n_group,
        # and topk_group (non-homogeneous)
        assert_homogeneous_moe(moe_layers, self._attrs)

        sample_module = next(iter(moe_layers.values()))
        self._num_experts = get_num_experts(sample_module, self._attrs)
        self._n_group = get_router_num_groups(sample_module, self._attrs)
        self._n_experts_to_drop = int(self._num_experts * self.sparsity)

        if self._n_experts_to_drop == 0:
            logger.warning(
                f"sparsity={self.sparsity} results in 0 "
                f"experts to drop (out of {self._num_experts}). No pruning will "
                "be performed."
            )
        else:
            # fail fast (before calibration) if the requested sparsity would
            # leave the router unable to select top_k experts per token
            assert_routing_feasible(
                self._num_experts,
                self._n_experts_to_drop,
                self._n_group,
                get_router_topk_group(sample_module, self._attrs),
                self._top_k,
            )

        logger.info(
            f"REAP initialized: {len(moe_layers)} MoE layers, "
            f"{self._num_experts} experts/layer, will drop "
            f"{self._n_experts_to_drop} ({self.sparsity:.0%})"
            + (f", n_group={self._n_group}" if self._n_group > 1 else "")
        )

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        self.started_ = True
        model = state.model

        if not get_calibrate_all_experts_flag():
            raise RuntimeError(
                "REAP requires that all experts be activated during calibration,"
                " but calibrate_all_experts is false. "
                "Ensure the model runs inside moe_calibration_context by setting "
                "dataset_args.moe_calibrate_all_experts in oneshot."
            )

        for layer_name in self._moe_layer_names:
            module = model.get_submodule(layer_name)

            self._saliency_trackers[layer_name] = REAPSaliencyTracker(self._num_experts)
            self._norm_buffers[layer_name] = {}

            # capture the router's raw output so routing is read from the model
            # itself rather than recomputed (correct for every routing scheme)
            router = getattr(module, self._attrs.router_attr)
            self.register_hook(
                router, partial(self._router_hook, layer_name), "forward"
            )

            # one hook per expert to record its per-token output norm
            experts = getattr(module, self._attrs.experts_attr)
            n_expert_hooks = 0
            for idx, expert in enumerate(experts.children()):
                self.register_hook(
                    expert, partial(self._expert_hook, layer_name, idx), "forward"
                )
                n_expert_hooks += 1
            if n_expert_hooks == 0:
                raise RuntimeError(
                    f"REAP could not register per-expert hooks for '{layer_name}': "
                    f"the experts module ({experts.__class__.__name__}) has no "
                    "child modules. Fused experts are not supported."
                )

            self.register_hook(
                module, partial(self._moe_post_hook, layer_name), "forward"
            )

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True
        # fallback for pipelines that never fire SEQUENTIAL_EPOCH_END
        self.on_sequential_epoch_end()
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        # Apply structural pruning here (after the calibration context has
        # exited): the decisions were finalized at SEQUENTIAL_EPOCH_END, so this
        # only mutates module structure -- it does not need calibration data.
        if not self.ended_:
            self.on_calibration_end(state, None)

        if self._n_experts_to_drop == 0:
            logger.info("REAP: nothing to prune (n_experts_to_drop=0)")
            return True

        missing = [n for n in self._moe_layer_names if n not in self._prune_decisions]
        if missing:
            raise RuntimeError(
                f"REAP did not finalize prune decisions for {len(missing)} MoE "
                f"layers (e.g. {missing[:3]}); no calibration data reached them."
            )

        model = state.model
        for layer_name, retained in self._prune_decisions.items():
            logger.debug(
                f"Pruning {layer_name}: keeping {len(retained)} experts {retained}"
            )
            prune_moe_layer(model, layer_name, retained, self._attrs)

        sample_module = model.get_submodule(self._moe_layer_names[0])
        new_num_experts = get_num_experts(sample_module, self._attrs)
        update_model_config(model, self._attrs, new_num_experts)

        self._prune_decisions.clear()
        self._saliency_trackers.clear()
        self._norm_buffers.clear()
        self._routing_cache.clear()

        return True

    # -- decision finalization ----------------------------------------------

    def on_sequential_epoch_end(self):
        """Finalize drop decisions for any tracked layer whose saliency is
        complete, then release its activation buffers."""
        if self._n_experts_to_drop == 0:
            return

        for layer_name, tracker in list(self._saliency_trackers.items()):
            if layer_name in self._prune_decisions:
                continue
            if tracker.total_count <= 0:
                continue

            retained = compute_retained_experts(
                tracker.mean_saliency, self._n_experts_to_drop, self._n_group
            )
            self._prune_decisions[layer_name] = retained

            # free this layer's accumulators / buffers now
            del self._saliency_trackers[layer_name]
            self._norm_buffers.pop(layer_name, None)
            self._routing_cache.pop(layer_name, None)

    # -- calibration hooks ---------------------------------------------------

    def _router_hook(
        self, layer_name: str, module: torch.nn.Module, args: tuple, output: Any
    ):
        """Cache the router's raw forward output for this layer/batch."""
        # hooks stay registered through the pipeline's error-propagation pass,
        # which re-runs an already-finalized subgraph; ignore those calls
        if layer_name not in self._saliency_trackers:
            return
        self._routing_cache[layer_name] = output

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
            return
        if isinstance(output, tuple):
            output = output[0]
        with torch.no_grad():
            norms = torch.linalg.norm(output.float(), dim=-1).reshape(-1)
        self._norm_buffers[layer_name][expert_idx] = norms

    def _moe_post_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Combine cached routing decisions with buffered expert norms."""
        tracker = self._saliency_trackers.get(layer_name)
        router_output = self._routing_cache.get(layer_name)
        if tracker is None or router_output is None:
            return

        norm_buffer = self._norm_buffers[layer_name]
        if len(norm_buffer) != tracker.num_experts:
            logger.warning(
                f"REAP: layer '{layer_name}' saw {len(norm_buffer)}/"
                f"{tracker.num_experts} experts this batch; skipping update."
            )
            self._norm_buffers[layer_name] = {}
            self._routing_cache[layer_name] = None
            return

        with torch.no_grad():
            topk_indices, topk_weights = extract_routing(
                router_output, module, self._attrs, self._top_k
            )
            expert_norms = torch.stack(
                [norm_buffer[i] for i in range(tracker.num_experts)], dim=1
            )
            tracker.update(topk_indices, topk_weights, expert_norms)

        self._norm_buffers[layer_name] = {}
        self._routing_cache[layer_name] = None
