import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MoE as OriginalDeepseekV3MoE,
)

from llmcompressor.modeling.moe_context import (
    MoECalibrationModule,
    register_moe_calibration,
)


@register_moe_calibration("DeepseekV3MoE")
class CalibrationDeepseekV3MoE(MoECalibrationModule):
    """
    Calibration version of DeepseekV3MoE that sends all tokens to all experts.
    """

    is_permanent = True

    def __init__(
        self,
        original: OriginalDeepseekV3MoE,
        config: DeepseekV3Config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.experts = original.experts
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Begin MoE
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx, expert in enumerate(self.experts):
            token_indices, weight_indices = torch.where(expert_mask[expert_idx])
            has_tokens = token_indices.numel() > 0

            if self.calibrate_all_experts:
                expert_input = hidden_states
                expert_output = expert(expert_input)

                if has_tokens:
                    expert_weights = topk_weights[token_indices, weight_indices]
                    routed_output = expert_output[
                        token_indices
                    ] * expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(0, token_indices, routed_output)
            else:
                # Normal MoE: only process tokens routed to this expert
                if has_tokens:
                    expert_input = hidden_states[token_indices]
                    expert_output = expert(expert_input)
                    expert_weights = topk_weights[token_indices, weight_indices]
                    routed_output = expert_output * expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(0, token_indices, routed_output)
        # End MoE

        hidden_states = final_hidden_states.type(hidden_states.dtype).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


# Legacy function for backward compatibility
def replace(
    config: DeepseekV3Config,
    module: OriginalDeepseekV3MoE,
    calibrate_all_experts: bool,
):
    """
    Legacy replacement function.
    Use CalibrationDeepseekV3MoE instead.
    """
    return CalibrationDeepseekV3MoE(
        module,
        config,
        calibrate_all_experts=calibrate_all_experts,
    )
