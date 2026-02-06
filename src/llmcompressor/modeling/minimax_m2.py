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
    """
    Calibration version of MiniMaxM2SparseMoeBlock that can send all tokens
    to all experts during calibration.

    When `calibrate_all_experts=True`, each expert is executed on all tokens so
    quantization statistics are collected for every expert, while routed-token
    weighting is still used for the final output.
    """

    is_permanent = False

    def __init__(
        self,
        original: MiniMaxM2SparseMoeBlock,
        config: MiniMaxM2Config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.experts = original.experts
        self.gate = original.gate
        self.calibrate_all_experts = calibrate_all_experts
        self.jitter_noise = original.jitter_noise
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
        _router_logits, top_k_weights, top_k_index = self.gate(
            hidden_states, self.e_score_correction_bias
        )

        if not self.calibrate_all_experts:
            final_hidden_states = self.experts(
                hidden_states, top_k_index, top_k_weights
            )
            return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        # Reimplementing MiniMaxM2Experts.forward only when calibrating all experts.
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = F.one_hot(top_k_index, num_classes=self.experts.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.experts.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            has_tokens = token_idx.numel() > 0

            # Run all tokens through the expert to gather stats.
            gate, up = F.linear(
                hidden_states, self.experts.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            expert_out_full = self.experts.act_fn(gate) * up
            expert_out_full = F.linear(
                expert_out_full, self.experts.down_proj[expert_idx]
            )
            if not has_tokens:
                continue
            expert_out = expert_out_full[token_idx]

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

        return final_hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original
