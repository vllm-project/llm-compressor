from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )


@MoECalibrationModule.register("Qwen3_5MoeSparseMoeBlock")
class CalibrationQwen3_5MoeSparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of Qwen3_5MoeSparseMoeBlock that sends all tokens to all
    experts. During calibration, when calibrate_all_experts=True, all tokens are
    sent to all experts to ensure proper quantization statistics are collected for
    every expert, not just those activated by the calibration data routing.

    The fused 3D expert tensors are permanently decomposed into individual
    Qwen3_5MoeMLP modules for vLLM serving compatibility.
    """

    is_permanent = True

    def __init__(
        self,
        original: Qwen3_5MoeSparseMoeBlock,
        config: Qwen3_5MoeConfig,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        text_config: Qwen3_5MoeTextConfig = config.get_text_config()

        self.hidden_size = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = text_config.num_experts_per_tok

        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.experts = SequentialQwen3_5MoeExperts(text_config, original.experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        # Qwen3_5MoeTopKRouter returns (router_logits, router_scores, router_indices)
        # but its gate.weight is an nn.Parameter, so we replicate the routing logic
        # here for consistency with other calibration modules in the repo.
        router_logits = torch.nn.functional.linear(
            hidden_states, self.gate.weight
        )
        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=-1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, token_idx = torch.where(expert_mask[expert_idx].squeeze(0))

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states)[token_idx]
            else:
                expert_out = expert_layer(hidden_states[token_idx])

            if len(token_idx) > 0:
                weighted_output = expert_out * routing_weights[token_idx, idx, None]
                final_hidden_states.index_add_(
                    0, token_idx, weighted_output.to(hidden_states.dtype)
                )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            torch.nn.functional.sigmoid(self.shared_expert_gate(hidden_states))
            * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original


class SequentialQwen3_5MoeExperts(torch.nn.ModuleList):
    """Decomposes fused 3D expert tensors into individual Qwen3_5MoeMLP modules."""

    def __init__(self, config, original):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeMLP,
        )

        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size

        with skip_weights_initialize():
            super().__init__(
                [
                    Qwen3_5MoeMLP(config, intermediate_size=intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        # Qwen3.5 gate_up_proj shape: [num_experts, 2*intermediate_dim, hidden_dim]
        # nn.Linear weight shape: [out_features, in_features]
        # F.linear(input, weight) = input @ weight.T — same convention as nn.Linear
        # So the 3D tensor slices map directly without transposing.
        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]
            down = original.down_proj[i]

            gate_proj = gate_up[:intermediate_size, :]
            up_proj = gate_up[intermediate_size:, :]

            self[i].gate_proj.weight.data = gate_proj.clone().contiguous()
            self[i].up_proj.weight.data = up_proj.clone().contiguous()
            self[i].down_proj.weight.data = down.clone().contiguous()
