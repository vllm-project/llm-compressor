"""
Utilities for MoNE pruning: calibration statistics, ranking, and novice
replacement.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from compressed_tensors import align_module_device
from loguru import logger

from llmcompressor.modeling.moe.linear_experts import (
    NoviceExpertMLP,
)
from llmcompressor.modifiers.pruning.reap.utils import MoeModelAttrs

__all__ = [
    "MoNEStatsTracker",
    "replace_experts_with_novices",
    "update_mone_model_config",
]


@dataclass
class MoNESelection:
    preserved: list[int]
    novices: list[int]


class MoNEStatsTracker:
    """
    Tracks the calibration statistics used by MoNE.

    For the main ``fusion`` ranking, MoNE keeps experts with large output
    fluctuation and large router access score. The output mean is retained as
    the novice constant for experts that are not preserved.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        ranking_metric: str,
        stats_device: str | None,
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.ranking_metric = ranking_metric
        self.stats_device = torch.device(stats_device) if stats_device else None

        self.num_tokens = torch.zeros(num_experts, dtype=torch.long)
        self.baseline_out: torch.Tensor | None = None
        self.fluc_out: torch.Tensor | None = None
        self.routing_sum: torch.Tensor | None = None
        self.routing_count: int = 0

    @property
    def needs_output_stats(self) -> bool:
        # Novice experts are initialized from expert output means regardless of
        # which statistic is used to rank experts.
        return True

    @property
    def needs_routing_stats(self) -> bool:
        return self.ranking_metric in ("routing_score", "fusion")

    def _device_for(self, tensor: torch.Tensor) -> torch.device:
        return self.stats_device or tensor.device

    def _ensure_output_stats(self, tensor: torch.Tensor):
        if self.baseline_out is not None:
            return

        device = self._device_for(tensor)
        self.baseline_out = torch.zeros(
            self.num_experts,
            self.hidden_size,
            dtype=torch.float32,
            device=device,
        )
        self.fluc_out = torch.zeros_like(self.baseline_out)

    def _ensure_routing_stats(self, tensor: torch.Tensor):
        if self.routing_sum is not None:
            return

        device = self._device_for(tensor)
        self.routing_sum = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            device=device,
        )

    @torch.no_grad()
    def update_expert(
        self,
        expert_idx: int,
        output: torch.Tensor,
    ):
        if not self.needs_output_stats:
            return

        if isinstance(output, tuple):
            output = output[0]

        output = output.reshape(-1, output.shape[-1])
        token_size = output.shape[0]
        if token_size == 0:
            return

        self._ensure_output_stats(output)

        old_tokens = int(self.num_tokens[expert_idx].item())
        new_tokens = old_tokens + token_size
        out_float = output.float()

        baseline = self.baseline_out[expert_idx]
        fluc = self.fluc_out[expert_idx]
        batch_mean = out_float.mean(dim=0).to(baseline.device)
        batch_m2 = torch.sum(
            (out_float - batch_mean.to(out_float.device).unsqueeze(0)).pow(2),
            dim=0,
        ).to(fluc.device)

        if old_tokens == 0:
            baseline.copy_(batch_mean)
            fluc.copy_(batch_m2 / token_size)
        else:
            old_mean = baseline.clone()
            delta = batch_mean - old_mean
            baseline.copy_(old_mean + delta * (token_size / new_tokens))

            old_m2 = fluc * old_tokens
            mean_shift_m2 = delta.to(fluc.device).pow(2) * (
                old_tokens * token_size / new_tokens
            )
            fluc.copy_((old_m2 + batch_m2 + mean_shift_m2) / new_tokens)

        self.num_tokens[expert_idx] = new_tokens

    @torch.no_grad()
    def update_routing(
        self,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        if not self.needs_routing_stats:
            return

        self._ensure_routing_stats(topk_weights)

        topk_indices = topk_indices.reshape(-1, topk_indices.shape[-1]).to(torch.long)
        topk_weights = topk_weights.reshape(topk_indices.shape).float()

        routing_weights = torch.zeros(
            topk_weights.shape[0],
            self.num_experts,
            dtype=torch.float32,
            device=topk_weights.device,
        )
        topk_weights = topk_weights.to(routing_weights.dtype)
        routing_weights = torch.scatter(
            routing_weights,
            dim=1,
            index=topk_indices,
            src=topk_weights,
        )

        # Average over all routed tokens, with zero contribution when an expert is
        # not selected. This keeps the routing score frequency-weighted:
        # E[gate_weight * 1(expert in top-k)].
        batch_size = routing_weights.shape[0]
        count = torch.full_like(self.routing_sum, float(self.routing_count))
        new_count = count + batch_size
        routing_sum = torch.sum(routing_weights, dim=0).to(self.routing_sum.device)
        self.routing_sum.mul_(count / new_count)
        self.routing_sum.add_(routing_sum / new_count)
        self.routing_count += batch_size

    @property
    def has_stats(self) -> bool:
        has_output = not self.needs_output_stats or self.num_tokens.sum().item() > 0
        has_routing = not self.needs_routing_stats or self.routing_count > 0
        return has_output and has_routing

    @property
    def mean_outputs(self) -> torch.Tensor:
        if self.baseline_out is None:
            return torch.zeros(
                self.num_experts,
                self.hidden_size,
                dtype=torch.float32,
            )
        return self.baseline_out.detach().cpu()

    @property
    def output_fluctuation(self) -> torch.Tensor:
        if self.fluc_out is None:
            return torch.zeros(self.num_experts, dtype=torch.float32)

        fluc = self.fluc_out.detach()
        if self.ranking_metric == "fusion":
            fluc = torch.sqrt(fluc.clamp_min(0))

        return torch.norm(fluc, dim=1)

    @property
    def routing_score(self) -> torch.Tensor:
        if self.routing_sum is None or self.routing_count <= 0:
            return torch.zeros(self.num_experts, dtype=torch.float32)
        return self.routing_sum.detach()

    @property
    def importance(self) -> torch.Tensor:
        if self.ranking_metric == "routing_score":
            return self.routing_score

        if self.ranking_metric == "output_fluctuation":
            return self.output_fluctuation

        if self.ranking_metric == "fusion":
            return self.output_fluctuation * self.routing_score

        raise ValueError(f"Unknown MoNE ranking metric: {self.ranking_metric}")

    def select_layerwise(self, preserve_n_experts: int) -> MoNESelection:
        scores = torch.nan_to_num(self.importance, nan=-float("inf"))
        preserve_n = min(preserve_n_experts, self.num_experts)

        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        if preserve_n > 0:
            mask[torch.topk(scores, preserve_n, largest=True).indices] = True

        preserved = [idx for idx, keep in enumerate(mask.tolist()) if keep]
        novices = [idx for idx, keep in enumerate(mask.tolist()) if not keep]
        return MoNESelection(preserved=preserved, novices=novices)


def replace_experts_with_novices(
    model: nn.Module,
    layer_name: str,
    novice_indices: list[int],
    mean_outputs: torch.Tensor,
    moe_attrs: MoeModelAttrs,
    zero_out_novice: bool,
) -> list[int]:
    moe_block = model.get_submodule(layer_name)
    experts = getattr(moe_block, moe_attrs.experts_attr)

    for expert_idx in novice_indices:
        old_expert = experts[expert_idx]
        hidden_size = mean_outputs.shape[-1]

        with align_module_device(old_expert):
            dtype = _first_floating_dtype(old_expert, mean_outputs.dtype)
            device = _first_device(old_expert, mean_outputs.device)
            approx_value = (
                torch.zeros(hidden_size, dtype=torch.float32)
                if zero_out_novice
                else mean_outputs[expert_idx]
            )

        novice = NoviceExpertMLP(
            hidden_dim=hidden_size,
            dtype=dtype,
        ).to(device)
        approx_value = approx_value.to(dtype=dtype)
        novice._llmcompressor_mone_approx_value_cpu = (
            approx_value.detach().cpu().clone()
        )
        with torch.no_grad():
            novice.approx_value.copy_(
                approx_value.to(device=novice.approx_value.device)
            )

        experts[expert_idx] = novice

    return novice_indices


def update_mone_model_config(
    model: nn.Module,
    moe_attrs: MoeModelAttrs,
    approximate_experts: dict[str, list[int]],
    implementation_metadata: dict | None = None,
):
    config = model.config.text_config if moe_attrs.has_text_config else model.config
    config.approximate_experts = approximate_experts
    if implementation_metadata is not None:
        config.llmcompressor_mone_implementation = implementation_metadata
    logger.info(
        f"Updated {config.__class__.__name__}.approximate_experts for "
        f"{len(approximate_experts)} MoE layers"
    )


def _first_device(module: nn.Module, fallback: torch.device) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return fallback


def _first_floating_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    for param in module.parameters(recurse=True):
        if param.is_floating_point():
            return param.dtype
    for buffer in module.buffers(recurse=True):
        if buffer.is_floating_point():
            return buffer.dtype
    return fallback
