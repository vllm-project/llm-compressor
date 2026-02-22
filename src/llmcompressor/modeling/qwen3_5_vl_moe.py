import torch
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize
import torch.nn.functional as F


@MoECalibrationModule.register("Qwen3_5MoeSparseMoeBlock")
class CalibrateQwen3_5MoeTextSparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of Qwen3_5MoeSparseMoeBlock that sends all tokens to all
    experts.
    """

    is_permanent = True

    def __init__(
        self,
        original: "Qwen3_5MoeSparseMoeBlock",
        config: "Qwen3_5MoeConfig",
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        text_config: "Qwen3_5MoeTextConfig" = config.get_text_config()

        self.num_experts = text_config.num_experts

        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.gate = original.gate
        self.experts = SequentialQwen3VLMoeTextExperts(text_config, original.experts)
        self.calibrate_all_experts = calibrate_all_experts
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # router: returns (router_logits, router_scores, router_indices)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

        # expert mask: (num_experts, top_k, num_tokens)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(
            2, 1, 0
        )

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, token_idx = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states_reshaped)[token_idx]
            else:
                expert_out = expert_layer(hidden_states_reshaped[token_idx])

            if len(token_idx) > 0:
                current_hidden_states = (
                    expert_out * routing_weights[token_idx, idx, None]
                )
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(hidden_states.dtype),
                )

        # shared expert
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            * shared_expert_output
        )
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original


class SequentialQwen3VLMoeTextExperts(torch.nn.ModuleList):
    def __init__(self, config, original):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeMLP,
        )
        from compressed_tensors.offload import disable_onloading

        self.num_experts = original.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__(
                [
                    Qwen3_5MoeMLP(
                        config, intermediate_size=config.shared_expert_intermediate_size
                    )
                    for _ in range(self.num_experts)
                ]
            )

        intermediate_size = original.down_proj.shape[-1]

        with disable_onloading():
            gate_up_data = original.gate_up_proj.data  # [num_experts, 2*inter, hidden]
            down_data = original.down_proj.data  # [num_experts, hidden, inter]

        for i in range(self.num_experts):
            gate_up = gate_up_data[i]
            down = down_data[i]

            gate_proj = gate_up[:intermediate_size, :]
            up_proj = gate_up[intermediate_size:, :]

            self[i].gate_proj.weight.data = gate_proj.clone().contiguous()
            self[i].up_proj.weight.data = up_proj.clone().contiguous()
            self[i].down_proj.weight.data = down.clone().contiguous()