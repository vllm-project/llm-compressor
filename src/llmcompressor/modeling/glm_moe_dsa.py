from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule

if TYPE_CHECKING:
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaMoE,
        GlmMoeDsaNaiveMoe,
    )

from llmcompressor.utils.dev import skip_weights_initialize


@MoECalibrationModule.register("GlmMoeDsaMoE")
class CalibrationGlmMoeDsaMoE(MoECalibrationModule):
    """
    Calibration version of GlmMoeDsaMoE that unpacks experts for sequential
    processing.

    This module:
    1. Unpacks the packed expert weights (3D -> 2D) for calibration
    2. Optionally sends all tokens to all experts during calibration
    3. Stays in unpacked form (permanent) for vLLM compatibility

    Subclasses (e.g. :class:`CalibrationGlm4MoeLiteMoE`) override
    :meth:`_get_num_experts` and :meth:`_make_experts` to handle
    model-specific config fields and MLP classes, while inheriting the
    shared routing and forward logic.
    """

    is_permanent = True

    def _get_num_experts(self, config) -> int:
        """Return the number of routed experts from the model config.

        Override in subclasses whose config stores the expert count under a
        different attribute name (e.g. ``n_routed_experts``).
        """
        return config.num_local_experts

    def _make_experts(self, config, original_experts) -> torch.nn.ModuleList:
        """Create the sequential (unpacked) expert module list.

        Override in subclasses that need a different MLP class for unpacking
        (e.g. ``Glm4MoeLiteMLP`` instead of ``GlmMoeDsaMLP``).
        """
        return SequentialGlmMoeDsaExperts(config, original_experts)

    def __init__(
        self,
        original: GlmMoeDsaMoE,
        config: GlmMoeDsaConfig,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = self._get_num_experts(config)
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.experts = self._make_experts(config, original.experts)
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                topk_indices, num_classes=self.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)

        for i in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[i])
            has_tokens = token_idx.numel() > 0

            if self.calibrate_all_experts:
                expert_out_all = self.experts[i](hidden_states)
                if not has_tokens:
                    continue
                expert_out = expert_out_all[token_idx]
            else:
                if not has_tokens:
                    continue
                expert_out = self.experts[i](hidden_states[token_idx])

            weighted_output = expert_out * topk_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, weighted_output.to(final_hidden_states.dtype)
            )

        hidden_states = final_hidden_states.type(hidden_states.dtype).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class SequentialGlmMoeDsaExperts(torch.nn.ModuleList):
    def __init__(self, config: GlmMoeDsaConfig, original: GlmMoeDsaNaiveMoe):
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaMLP

        self.num_experts = config.num_local_experts
        with skip_weights_initialize():
            super().__init__(
                [
                    GlmMoeDsaMLP(config, intermediate_size=config.moe_intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]
            down = original.down_proj[i]

            gate_proj, up_proj = gate_up.chunk(2, dim=0)

            self[i].gate_proj.weight.data = gate_proj.clone().contiguous()
            self[i].up_proj.weight.data = up_proj.clone().contiguous()
            self[i].down_proj.weight.data = down.clone().contiguous()
