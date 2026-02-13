import torch
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeMoE as OriginalGlm4MoeMoE,
)

from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("Glm4MoeMoE")
class CalibrationGlm4MoeMoE(MoECalibrationModule):
    """
    Calibration version of Glm4MoeMoE that sends all tokens to all experts.
    During calibration, when calibrate_all_experts=True, all tokens are sent to
    all experts to ensure proper quantization statistics are collected for every
    expert, not just those activated by the calibration data routing.
    """

    is_permanent = False

    def __init__(
        self,
        original: OriginalGlm4MoeMoE,
        config: Glm4MoeConfig,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.experts = original.experts
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional calibration mode.
        When calibrate_all_experts=True:
            - All tokens are sent to all experts for calibration
            - Routing weights are still used for final output combination
            - This ensures all experts see calibration data
        When calibrate_all_experts=False:
            - Normal MoE routing behavior (only routed tokens go to each expert)
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Begin MoE - inline the moe() method logic with calibration support
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx, expert in enumerate(self.experts):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            has_tokens = token_indices.numel() > 0

            if self.calibrate_all_experts:
                # When calibrating, run all tokens through the expert to gather stats.
                # The output is still calculated using only the routed tokens.
                expert_output_full = expert(hidden_states)
                if not has_tokens:
                    # No tokens routed to this expert, but stats were gathered.
                    continue
                expert_output = expert_output_full[token_indices]
            else:
                # Standard MoE behavior: only process tokens routed to this expert.
                if not has_tokens:
                    continue
                expert_output = expert(hidden_states[token_indices])

            # Common logic for combining expert outputs
            expert_weights = topk_weights[token_indices, weight_indices]
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, weighted_output)
        # End MoE

        hidden_states = final_hidden_states.type(hidden_states.dtype).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        """
        Restore the original module structure.

        Since is_permanent=False, this method is called when exiting
        the calibration context to restore the original MoE module.
        """
        return original
