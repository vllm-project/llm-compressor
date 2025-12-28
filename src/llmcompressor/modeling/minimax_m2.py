import torch
import torch.nn as nn

from llmcompressor.transformers.moe.calibration import (
    MoECalibrationModule,
    register_moe_calibration,
)

@register_moe_calibration("MiniMaxM2SparseMoeBlock")
class CalibrationMiniMaxM2SparseMoeBlock(MoECalibrationModule):
    """
    Calibration module for MiniMaxM2SparseMoeBlock that supports calibrating all experts.
    """

    # Non-permanent replacement: restored to original module after calibration context
    is_permanent = False

    def __init__(
        self,
        original: MiniMaxM2SparseMoeBlock,
        config: MiniMaxM2Config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.original = original
        self.num_experts = config.num_experts

        # gating
        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate
        self.experts = original.experts
        self.route_tokens_to_experts = original.route_tokens_to_experts

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.calibrate_all_experts:
            return self.original(hidden_states)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # 1. Router Logic
        router_logits = self.gate(hidden_states_reshaped)
        top_k_index, top_k_weights = self.route_tokens_to_experts(
            router_logits
        )

        # 2. Calibrate All Experts
        # Run every expert on the full input to collect statistics for all tokens.
        final_hidden_states = torch.zeros_like(hidden_states_reshaped)

        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx in range(num_experts):
            # Run expert on ALL tokens for calibration statistics
            expert_output = self.original.experts[expert_idx](hidden_states_reshaped)

            # Select only routed outputs for the final result
            # to keep downstream activations valid.
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden
            # state for the current expert. We need to make sure to multiply
            # the output hidden states by `routing_weights` on the
            # corresponding tokens (top-1 and top-2)
            if len(top_x) > 0:
                current_hidden_states = expert_output[top_x] * top_k_weights[top_x, idx, None]
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        return final_hidden_states, router_logits

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return self.original