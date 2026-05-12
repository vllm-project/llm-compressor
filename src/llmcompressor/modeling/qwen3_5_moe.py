from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )


@MoECalibrationModule.register("Qwen3_5MoeSparseMoeBlock")
class CalibrationQwen3_5MoeSparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of Qwen3_5MoeSparseMoeBlock that unfuses 3D expert
    parameters into individual MLP modules (nn.Linear) so they can be
    individually quantized. Sends all tokens to all experts during calibration.

    is_permanent = True because the unfused structure must persist for
    quantization to target the individual nn.Linear expert weights.
    """

    is_permanent = True

    def __init__(
        self,
        original: Qwen3_5MoeSparseMoeBlock,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        text_config = getattr(config, "text_config", config)

        self.calibrate_all_experts = calibrate_all_experts

        # Use plain Linear for gate so module_type() returns "Linear"
        # This ensures gates appear in the ignore list when config is saved
        original_weight = original.gate.weight.data
        self.gate = torch.nn.Linear(
            text_config.hidden_size, text_config.num_experts, bias=False
        )
        self.gate.weight.data = self.gate.weight.data.to(
            dtype=original_weight.dtype, device=original_weight.device
        )
        self.gate.weight.data.copy_(original_weight)

        # Store routing parameters needed for forward pass
        self.top_k = text_config.num_experts_per_tok
        self.num_experts = text_config.num_experts
        self.hidden_dim = text_config.hidden_size
        self.hidden_size = text_config.hidden_size

        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.experts = SequentialQwen3_5MoeExperts(text_config, original.experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # Perform routing (previously in Qwen3VLMoeTextTopKRouter.forward)
        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)

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
        return self


class SequentialQwen3_5MoeExperts(torch.nn.ModuleList):
    """
    Unfuses 3D expert parameter tensors into individual Qwen3_5MoeMLP modules
    so that each expert's weights are nn.Linear and can be targeted by
    quantization with targets="Linear".
    """

    def __init__(self, config, original):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeMLP,
        )

        self.num_experts = config.num_experts
        intermediate_size = config.moe_intermediate_size

        with skip_weights_initialize():
            super().__init__(
                [
                    Qwen3_5MoeMLP(config, intermediate_size=intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        gate_up_data = original.gate_up_proj.data  # [num_experts, 2*inter, hidden]
        down_data = original.down_proj.data  # [num_experts, hidden, inter]

        for i in range(self.num_experts):
            gate_up = gate_up_data[i]  # [2*intermediate, hidden]
            down = down_data[i]  # [hidden, intermediate]

            # gate_up_proj stores [gate; up] stacked along dim 0
            # nn.Linear weight is [out_features, in_features]
            self[i].gate_proj.weight.data = (
                gate_up[:intermediate_size, :].clone().contiguous()
            )
            self[i].up_proj.weight.data = (
                gate_up[intermediate_size:, :].clone().contiguous()
            )
            self[i].down_proj.weight.data = down.clone().contiguous()
