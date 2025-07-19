import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MoE as OriginalDeepseekV3MoE,
)


class DeepseekV3MoECalibrate(torch.nn.Module):
    """
    Patched DeepseekV3MoE which sends all tokens to all experts for calibration
    """

    def __init__(self, config: DeepseekV3Config, original: OriginalDeepseekV3MoE):
        super().__init__()
        self.config = config
        self.experts = original.experts
        self.gate = original.gate
        self.shared_experts = original.shared_experts

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

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            expert_weights = topk_weights[token_indices, weight_indices]
            expert_input = hidden_states[token_indices]
            expert_output = expert(expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            if token_indices.numel() > 0:
                final_hidden_states.index_add_(0, token_indices, weighted_output)
        # End MoE

        hidden_states = final_hidden_states.type(hidden_states.dtype).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


def replace(config: DeepseekV3Config, module: OriginalDeepseekV3MoE):
    return DeepseekV3MoECalibrate(config=config, original=module)
