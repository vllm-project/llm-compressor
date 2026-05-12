"""
REAP (Router-weighted Expert Activation Pruning) modifier for MoE models.

See: https://arxiv.org/abs/2510.13999
"""

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.reap.utils import (
    MoEModelAttrs,
    REAPSaliencyTracker,
    detect_moe_attrs,
    find_moe_layers,
    get_num_experts,
    prune_moe_layer,
    update_model_config,
)

__all__ = ["REAPPruningModifier"]


class REAPPruningModifier(Modifier):
    """
    Prunes experts from MoE layers using the REAP saliency metric:
    S_j = mean(g_j * ||f_j||_2), averaged over tokens routed to expert j.

    :param compression_ratio: fraction of experts to remove per layer (0, 1).
    :param ignore: module name patterns to skip during MoE layer detection.

    Example recipe::

        REAPPruningModifier:
          compression_ratio: 0.5
    """

    compression_ratio: float = 0.5
    ignore: list[str] = Field(default_factory=list)

    _attrs: MoEModelAttrs | None = PrivateAttr(default=None)
    _moe_layer_names: list[str] = PrivateAttr(default_factory=list)
    _saliency_trackers: dict[str, REAPSaliencyTracker] = PrivateAttr(
        default_factory=dict
    )
    _top_k: int = PrivateAttr(default=2)
    _n_experts_to_drop: int = PrivateAttr(default=0)
    _norm_buffers: dict[str, dict[int, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )
    _routing_cache: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_compression_ratio(self) -> "REAPPruningModifier":
        if not 0.0 < self.compression_ratio < 1.0:
            raise ValueError(
                f"compression_ratio must be in (0, 1), got {self.compression_ratio}"
            )
        return self

    def on_initialize(self, state: State, **kwargs) -> bool:
        model = state.model

        self._attrs = detect_moe_attrs(model)
        if self._attrs is None:
            raise ValueError(
                "Could not detect MoE architecture. Ensure the model is an "
                "MoE model or specify 'targets' explicitly."
            )

        moe_layers = find_moe_layers(model, self._attrs, self.ignore)
        if not moe_layers:
            raise ValueError("No MoE layers found in model.")
        self._moe_layer_names = list(moe_layers.keys())

        config = model.config
        text_config = getattr(config, "text_config", config)
        self._top_k = getattr(text_config, "num_experts_per_tok", 2)

        sample_module = next(iter(moe_layers.values()))
        num_experts = get_num_experts(sample_module, self._attrs)
        self._n_experts_to_drop = int(num_experts * self.compression_ratio)

        if self._n_experts_to_drop == 0:
            logger.warning(
                f"compression_ratio={self.compression_ratio} results in 0 "
                f"experts to drop (out of {num_experts}). No pruning will "
                "be performed."
            )

        logger.info(
            f"REAP initialized: {len(moe_layers)} MoE layers, "
            f"{num_experts} experts/layer, will drop "
            f"{self._n_experts_to_drop} ({self.compression_ratio:.0%})"
        )

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True
        model = state.model

        for layer_name in self._moe_layer_names:
            module = model.get_submodule(layer_name)
            num_experts = get_num_experts(module, self._attrs)
            self._saliency_trackers[layer_name] = REAPSaliencyTracker(num_experts)
            self._norm_buffers[layer_name] = {}

            if hasattr(module, "calibrate_all_experts"):
                module.calibrate_all_experts = True

            self.register_hook(
                module,
                partial(self._moe_pre_hook, layer_name),
                "forward_pre",
            )

            experts = getattr(module, self._attrs.experts_attr)
            for idx, expert in enumerate(experts.children()):
                self.register_hook(
                    expert,
                    partial(self._expert_hook, layer_name, idx),
                    "forward",
                )

            self.register_hook(
                module,
                partial(self._moe_post_hook, layer_name),
                "forward",
            )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        # Prune in on_finalize (not on_end) so we act on actual modules,
        # not calibration wrappers that get restored after the context exits.
        if not self.ended_:
            self.on_end(state, None)

        if self._n_experts_to_drop == 0:
            logger.info("REAP: nothing to prune (n_experts_to_drop=0)")
            return True

        model = state.model

        for layer_name in self._moe_layer_names:
            tracker = self._saliency_trackers[layer_name]
            saliency = tracker.mean_saliency

            logger.debug(f"REAP saliency for {layer_name}: {saliency.tolist()}")

            prune_moe_layer(
                model,
                layer_name,
                saliency,
                self._n_experts_to_drop,
                self._attrs,
            )

        sample_module = model.get_submodule(self._moe_layer_names[0])
        new_num_experts = get_num_experts(sample_module, self._attrs)
        update_model_config(model, self._attrs, new_num_experts)
        self._saliency_trackers.clear()

        return True

    def _moe_pre_hook(
        self, layer_name: str, module: torch.nn.Module, args: tuple
    ):
        """Run the router (one Linear layer) and cache routing decisions."""
        hidden_states = args[0]
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        router = getattr(module, self._attrs.router_attr)

        with torch.no_grad():
            router_output = router(hidden_states)
            if isinstance(router_output, tuple):
                router_logits = router_output[-1]
            else:
                router_logits = router_output

            routing_weights = F.softmax(
                router_logits, dim=-1, dtype=torch.float32
            )
            _, topk_indices = torch.topk(
                routing_weights, self._top_k, dim=-1
            )

        self._routing_cache[layer_name] = {
            "routing_weights": routing_weights,
            "topk_indices": topk_indices,
        }

    def _expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Buffer output norms from the calibration forward pass."""
        if isinstance(output, tuple):
            output = output[0]
        with torch.no_grad():
            norms = torch.linalg.norm(output.float(), dim=-1)
        self._norm_buffers[layer_name][expert_idx] = norms

    def _moe_post_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Combine cached routing decisions with buffered expert norms."""
        cache = self._routing_cache[layer_name]
        routing_weights = cache["routing_weights"]
        topk_indices = cache["topk_indices"]
        tracker = self._saliency_trackers[layer_name]

        with torch.no_grad():
            for expert_idx in range(tracker.num_experts):
                norms = self._norm_buffers[layer_name].get(expert_idx)
                if norms is None:
                    continue

                token_mask = (topk_indices == expert_idx).any(dim=-1)
                if not token_mask.any():
                    continue

                gate_vals = routing_weights[token_mask, expert_idx]
                routed_norms = norms[token_mask]
                tracker.update(expert_idx, gate_vals, routed_norms)

        self._norm_buffers[layer_name].clear()
        self._routing_cache[layer_name] = None
