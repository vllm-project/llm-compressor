"""
MoNE (Mixture of Novice Experts) pruning modifier for MoE models.

See: https://arxiv.org/abs/2507.00390
"""

from functools import partial
from typing import Any, Literal

import torch
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator

from llmcompressor.core import Event, State
from llmcompressor.core.session_functions import active_session
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.mone.utils import (
    MoNEStatsTracker,
    replace_experts_with_novices,
    update_mone_model_config,
)
from llmcompressor.modifiers.pruning.reap.utils import MoeModelAttrs, get_moe_attrs

__all__ = ["MoNEPruningModifier"]


class MoNEPruningModifier(Modifier):
    """
    Replaces low-importance MoE experts with lightweight novice experts.

    MoNE keeps the original router dimensionality and logical expert ids. Experts
    selected for pruning are replaced by a constant-output module whose value is
    the calibration-set mean output of the original expert.

    :param preserve_n_experts: number of full experts to preserve per layer when
        ``ranking_scope="layer"``.
    :param sparsity: optional fraction of experts to replace. Exactly one of
        ``preserve_n_experts`` and ``sparsity`` must be set.
    :param ranking_metric: expert importance metric. ``fusion`` matches the main
        MoNE recipe: output fluctuation multiplied by router access score.
    :param ranking_scope: ``layer`` preserves the same number of experts in each
        MoE layer; ``model`` applies a global budget across all tracked layers.
    :param fusion_io_weight: scalar used by the reference fusion score.
    :param zero_out_novice: replace pruned experts with zero-output novices
        instead of mean-output novices.
    :param stats_device: device for calibration statistics. Defaults to CPU to
        avoid holding all expert means/fluctuations on GPU.
    :param debug_path: optional path to a ``torch.save`` dump of MoNE scores and
        statistics, for implementation-equivalence debugging.
    :param ignore: module name patterns to skip during MoE layer detection.
    """

    preserve_n_experts: int | None = None
    sparsity: float | None = None
    ranking_metric: Literal["fusion", "routing_score", "output_fluctuation"] = "fusion"
    ranking_scope: Literal["layer", "model"] = "layer"
    fusion_io_weight: float = 0.5
    zero_out_novice: bool = False
    stats_device: str | None = "cpu"
    debug_path: str | None = None
    ignore: list[str] = Field(default_factory=list)

    _moe_attrs: MoeModelAttrs | None = PrivateAttr(default=None)
    _preserve_n_experts: int = PrivateAttr(default=0)
    _stats_trackers: dict[str, MoNEStatsTracker] = PrivateAttr(default_factory=dict)
    _applied: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _validate_config(self) -> "MoNEPruningModifier":
        if (self.preserve_n_experts is None) == (self.sparsity is None):
            raise ValueError(
                "Exactly one of preserve_n_experts or sparsity must be provided"
            )

        if self.preserve_n_experts is not None and self.preserve_n_experts < 0:
            raise ValueError(
                f"preserve_n_experts must be non-negative, got "
                f"{self.preserve_n_experts}"
            )

        if self.sparsity is not None and not 0.0 < self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in (0, 1), got {self.sparsity}")

        if not 0.0 <= self.fusion_io_weight <= 1.0:
            raise ValueError(
                f"fusion_io_weight must be in [0, 1], got {self.fusion_io_weight}"
            )

        return self

    def on_initialize(self, state: State, **kwargs) -> bool:
        model = state.model
        self._moe_attrs = get_moe_attrs(model, self.ignore)

        if self.sparsity is not None:
            n_experts_to_replace = int(self._moe_attrs.num_experts * self.sparsity)
            if n_experts_to_replace == 0:
                raise ValueError(
                    f"sparsity={self.sparsity} results in 0 experts to replace "
                    f"(out of {self._moe_attrs.num_experts}). "
                    "No pruning will be performed."
                )
            self._preserve_n_experts = (
                self._moe_attrs.num_experts - n_experts_to_replace
            )
        else:
            self._preserve_n_experts = self.preserve_n_experts

        if self._preserve_n_experts >= self._moe_attrs.num_experts:
            raise ValueError(
                f"preserve_n_experts={self._preserve_n_experts} must be smaller "
                f"than num_experts={self._moe_attrs.num_experts}"
            )

        logger.info(
            f"MoNE initialized: {len(self._moe_attrs.moe_layer_names)} MoE layers, "
            f"{self._moe_attrs.num_experts} experts/layer, preserving "
            f"{self._preserve_n_experts} full experts using "
            f"{self.ranking_metric} ranking"
        )

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        if get_calibrate_all_experts_flag():
            raise ValueError(
                "MoNEPruningModifier requires moe_calibrate_all_experts=False. "
                "MoNE's output means and fluctuations must be collected over "
                "routed tokens only to match the reference implementation."
            )

        session = active_session()
        recipe = getattr(session.lifecycle, "recipe", None)
        modifiers = getattr(recipe, "modifiers", None)
        if modifiers is not None and len(modifiers) > 1:
            raise ValueError(
                "MoNEPruningModifier must be the only modifier in the recipe "
                "during calibration. Other modifiers may interfere with MoNE "
                "statistics. Use the independent pipeline for multi-step flows."
            )

        model = state.model
        hidden_size = _hidden_size(model, self._moe_attrs)

        for layer_name in self._moe_attrs.moe_layer_names:
            module = model.get_submodule(layer_name)
            experts = getattr(module, self._moe_attrs.experts_attr)
            tracker = MoNEStatsTracker(
                num_experts=self._moe_attrs.num_experts,
                hidden_size=hidden_size,
                ranking_metric=self.ranking_metric,
                fusion_io_weight=self.fusion_io_weight,
                stats_device=self.stats_device,
            )
            self._stats_trackers[layer_name] = tracker

            if tracker.needs_output_stats:
                expert_list = [
                    expert
                    for expert in experts.children()
                    if isinstance(expert, ExpertMLP)
                ]
                for idx, expert in enumerate(expert_list):
                    self.register_hook(
                        expert,
                        partial(self._expert_hook, layer_name, idx),
                        "forward",
                    )

            if tracker.needs_routing_stats:
                self.register_hook(
                    experts,
                    partial(self._experts_block_hook, layer_name),
                    "forward",
                )

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.remove_hooks()
        self._apply_mone(state.model)

    def on_finalize(self, state: State, **kwargs) -> bool:
        self._stats_trackers.clear()
        return True

    def _expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        tracker = self._stats_trackers.get(layer_name)
        if tracker is None:
            return

        with torch.no_grad():
            tracker.update_expert(expert_idx, output)

    def _experts_block_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        tracker = self._stats_trackers.get(layer_name)
        if tracker is None:
            return

        top_k_indices = args[1]
        top_k_weights = args[2]
        with torch.no_grad():
            tracker.update_routing(top_k_indices, top_k_weights)

    def _apply_mone(self, model: torch.nn.Module):
        if self._applied:
            return

        approximate_experts = self._select_novices()
        self._dump_debug_stats(approximate_experts)

        for layer_name, novice_indices in approximate_experts.items():
            tracker = self._stats_trackers[layer_name]
            replace_experts_with_novices(
                model=model,
                layer_name=layer_name,
                novice_indices=novice_indices,
                mean_outputs=tracker.mean_outputs,
                moe_attrs=self._moe_attrs,
                zero_out_novice=self.zero_out_novice,
            )

        update_mone_model_config(
            model=model,
            moe_attrs=self._moe_attrs,
            approximate_experts=_config_keyed(approximate_experts),
            implementation_metadata={
                "algorithm": "mone",
            },
        )

        self._applied = True

    def _dump_debug_stats(
        self,
        approximate_experts: dict[str, list[int]],
    ):
        if self.debug_path is None:
            return

        from pathlib import Path

        debug_path = Path(self.debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "preserve_n_experts": self._preserve_n_experts,
                "ranking_metric": self.ranking_metric,
                "ranking_scope": self.ranking_scope,
                "fusion_io_weight": self.fusion_io_weight,
                "stats_device": self.stats_device,
                "approximate_experts": _config_keyed(approximate_experts),
                "layers": {
                    _config_layer_key(layer_name): {
                        "importance": tracker.importance.detach().cpu(),
                        "routing_score": tracker.routing_score.detach().cpu(),
                        "output_fluctuation": (
                            tracker.output_fluctuation.detach().cpu()
                        ),
                        "num_tokens": tracker.num_tokens.detach().cpu(),
                        "mean_outputs": tracker.mean_outputs,
                    }
                    for layer_name, tracker in self._stats_trackers.items()
                },
            },
            debug_path,
        )
        logger.info(f"Wrote MoNE debug stats to {debug_path}")

    def _select_novices(self) -> dict[str, list[int]]:
        for layer_name, tracker in self._stats_trackers.items():
            if not tracker.has_stats:
                raise ValueError(
                    f"MoNE did not collect calibration statistics for {layer_name}"
                )

        if self.ranking_scope == "layer":
            approximate_experts = {}
            for layer_name, tracker in self._stats_trackers.items():
                selection = tracker.select_layerwise(self._preserve_n_experts)
                approximate_experts[layer_name] = selection.novices
            return approximate_experts

        return self._select_novices_globally()

    def _select_novices_globally(
        self,
    ) -> dict[str, list[int]]:
        layer_names = list(self._stats_trackers.keys())
        scores = torch.cat(
            [
                torch.nan_to_num(
                    self._stats_trackers[layer_name].importance,
                    nan=-float("inf"),
                )
                for layer_name in layer_names
            ]
        )

        total_experts = scores.numel()
        total_preserved = min(
            self._preserve_n_experts * len(layer_names),
            total_experts,
        )
        preserved_flat = set()
        if total_preserved > 0:
            preserved_flat = set(
                int(idx)
                for idx in torch.topk(scores, total_preserved, largest=True)
                .indices.cpu()
                .tolist()
            )

        approximate_experts = {}
        offset = 0
        for layer_name in layer_names:
            tracker = self._stats_trackers[layer_name]
            novices = []
            for expert_idx in range(tracker.num_experts):
                if offset + expert_idx not in preserved_flat:
                    novices.append(expert_idx)
            approximate_experts[layer_name] = novices
            offset += tracker.num_experts

        return approximate_experts


def _hidden_size(model: torch.nn.Module, moe_attrs: MoeModelAttrs) -> int:
    config = model.config.text_config if moe_attrs.has_text_config else model.config
    for key in ("hidden_size", "hidden_dim"):
        if hasattr(config, key):
            return getattr(config, key)
    raise ValueError("Could not find hidden size in model config")


def _config_keyed(layer_values: dict[str, list[int]]) -> dict[str, list[int]]:
    return {
        _config_layer_key(layer_name): values
        for layer_name, values in layer_values.items()
    }


def _config_layer_key(layer_name: str) -> str:
    parts = layer_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers" and parts[idx + 1].isdigit():
            return parts[idx + 1]
    return layer_name
