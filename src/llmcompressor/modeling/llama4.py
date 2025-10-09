from typing import Tuple

import torch
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextExperts,
    Llama4TextMLP,
    Llama4TextMoe,
)

from llmcompressor.utils.dev import skip_weights_initialize


class SequentialLlama4TextMoe(torch.nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        original: Llama4TextMoe,
        calibrate_all_experts: bool,
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts

        self.experts = SequentialLlama4TextExperts(config, original.experts)
        self.router = original.router
        self.shared_expert = original.shared_expert
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_outputs = self.router(hidden_states)

        # support transformers 4.53 and greater
        if isinstance(router_outputs, tuple):
            router_scores, router_logits = router_outputs
        else:
            router_top_value, router_indices = torch.topk(
                router_logits, self.top_k, dim=1
            )
            router_logits = router_outputs
            router_scores = (
                torch.full_like(router_logits, float("-inf"))
                .scatter_(1, router_indices, router_top_value)
                .transpose(0, 1)
            )
            router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        out = self.shared_expert(hidden_states)
        for expert_index in range(self.num_experts):
            top_token_mask = router_scores[:, expert_index] > 0

            if self.calibrate_all_experts:
                # Run all tokens for calibration
                expert_out = self.experts[expert_index](hidden_states)[top_token_mask]
            else:
                expert_out = self.experts[expert_index](hidden_states[top_token_mask])

            # Only top-k tokens contribute to final output
            if top_token_mask.any():
                expert_score = router_scores[top_token_mask, expert_index].unsqueeze(-1)
                out[top_token_mask] += expert_out * expert_score

        return out, router_scores


class SequentialLlama4TextExperts(torch.nn.ModuleList):
    def __init__(self, config: Llama4TextConfig, original: Llama4TextExperts):
        self.num_experts = original.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])

        intermediate_size = original.down_proj.shape[1]

        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]
            down = original.down_proj[i]

            gate_proj = gate_up[:, :intermediate_size]
            up_proj = gate_up[:, intermediate_size:]

            self[i].gate_proj.weight.data = gate_proj.t().clone().contiguous()
            self[i].up_proj.weight.data = up_proj.t().clone().contiguous()
            self[i].down_proj.weight.data = down.t().clone().contiguous()


def replace(config: Llama4Config, module: Llama4TextMoe, calibrate_all_experts: bool):
    return SequentialLlama4TextMoe(
        config=config.get_text_config(),
        original=module,
        calibrate_all_experts=calibrate_all_experts,
    )
