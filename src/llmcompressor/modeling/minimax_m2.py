from __future__ import annotations

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
    """Calibration module for MiniMaxM2SparseMoeBlock with all-expert calibration."""

    is_permanent = False

    def __init__(
        self,
        original: MiniMaxM2SparseMoeBlock,
        config: MiniMaxM2Config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()

        # Gating
        self.calibrate_all_experts = calibrate_all_experts

        # Extract submodules directly to prevent parameter duplication
        # in find_tied_parameters (caused by holding self.original)
        self.gate = original.gate
        self.experts = original.experts

        # MiniMax specific parameters
        self.jitter_noise = original.jitter_noise
        self.num_experts = config.num_local_experts
        self.top_k = original.top_k
        # Use unbound function so this module's buffers are used.
        self._route_tokens_to_experts = type(original).route_tokens_to_experts
        self.register_buffer(
            "e_score_correction_bias", original.e_score_correction_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional all-expert calibration mode.

        - `calibrate_all_experts=False`: use upstream expert execution path.
        - `calibrate_all_experts=True`: execute every expert on all tokens,
          then aggregate only routed-token outputs.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        if self.e_score_correction_bias.device != router_logits.device:
            self.e_score_correction_bias = self.e_score_correction_bias.to(
                router_logits.device
            )
        top_k_index, top_k_weights = self._route_tokens_to_experts(self, router_logits)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states)[token_idx]
            else:
                expert_out = expert_layer(hidden_states[token_idx])

            if token_idx.numel() > 0:
                expert_weights = top_k_weights[token_idx, top_k_pos]
                weighted_output = expert_out * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    weighted_output.to(hidden_states.dtype),
                )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        return final_hidden_states, router_logits

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original
