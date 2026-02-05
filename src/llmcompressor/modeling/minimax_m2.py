from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from llmcompressor.modeling.moe_context import MoECalibrationModule

if TYPE_CHECKING:
    from transformers import MiniMaxM2Config
    from transformers.models.minimax_m2.modeling_minimax_m2 import (
        MiniMaxM2SparseMoeBlock,
    )


@MoECalibrationModule.register("MiniMaxM2SparseMoeBlock")
class CalibrationMiniMaxM2SparseMoeBlock(MoECalibrationModule):
    """
    Calibration module for MiniMaxM2SparseMoeBlock that supports calibrating all experts
    """

    is_permanent = False

    def __init__(
        self,
        original: MiniMaxM2SparseMoeBlock,
        config: MiniMaxM2Config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()

        # gating
        self.calibrate_all_experts = calibrate_all_experts

        # Extract submodules directly to prevent parameter duplication
        # in find_tied_parameters (caused by holding self.original)
        self.gate = original.gate
        self.experts = original.experts

        # MiniMax specific parameters
        self.jitter_noise = original.jitter_noise
        self.register_buffer(
            "e_score_correction_bias", original.e_score_correction_bias
        )
        self.route_tokens_to_experts = original.route_tokens_to_experts

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)

        # Reimplementing MiniMaxM2Experts.forward
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = F.one_hot(
            top_k_index, num_classes=self.experts.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the
        # computation on each expert
        # expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states)[top_x]
            else:
                expert_out = expert_layer(hidden_states[top_x])

            # Index the correct hidden states and compute the expert hidden
            # state for the current expert. We need to make sure to multiply
            # the output hidden states by `top_k_weights` on the
            # corresponding tokens (top-1 and top-2)
            if len(top_x) > 0:
                current_hidden_states = expert_out * top_k_weights[top_x, idx, None]
                final_hidden_states.index_add_(
                    0,
                    top_x,
                    current_hidden_states.to(hidden_states.dtype),
                )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        return final_hidden_states, router_logits

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original
