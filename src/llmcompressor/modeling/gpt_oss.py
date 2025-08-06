from typing import Tuple

import torch
from torch import nn
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssTopKRouter,
    GptOssExperts,
    GptOssMLP
)

from llmcompressor.utils.dev import skip_weights_initialize

class SequentialGptOssMLP(nn.Module):
    def __init__(self, config: GptOssMLP, original: GptOssMLP):
        super().__init__()
        self.router = original.router
        self.experts = SequentialGptOssExperts(config, original)

        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        # TODO: 
        #self.alpha = 1.702
        #self.limit = 7.0

    def forward(self, hidden_states):
        # original code : from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py
        routing_score, router_indices = self.router(hidden_states)
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_score.shape[1]
        if self.training:
            routing_weights = routing_score
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                # we sum on the top_k and on the sequence lenght to get which experts
                # are hit this time around
                expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hitted[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                out = self.experts[expert_idx](current_state)
                weighted_output = out * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            routing_weights = routing_score.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            for expert_idx in range(num_experts):
                out = self.experts[expert_idx](hidden_states)
                weighted_output = out * routing_weights[expert_idx, :, None]
                if expert_idx==0:
                    next_states = weighted_output
                else:
                    next_states = weighted_output.sum(dim=0)
        return next_states, routing_score

class SequentialGptOssExperts(torch.nn.ModuleList):
    def __init__(self, config, original):
        self.num_experts = config.num_local_experts
        with skip_weights_initialize():
            super().__init__([SequentialGptOssExpert(config) for _ in range(self.num_experts)])
        for i in range(self.num_experts):
            self[i].gate_proj.weight.data = original.experts.gate_up_proj[i,:,::2].t().clone().contiguous()
            self[i].gate_proj.bias.data = original.experts.gate_up_proj_bias[i,::2].t().clone().contiguous()
            self[i].up_proj.weight.data = original.experts.gate_up_proj[i,:,1::2].t().clone().contiguous()
            self[i].up_proj.bias.data = original.experts.gate_up_proj_bias[i,1::2].t().clone().contiguous()
            self[i].down_proj.weight.data = original.experts.down_proj[i,:,:].t().clone().contiguous()
            self[i].down_proj.bias.data = original.experts.down_proj_bias[i,:].t().clone().contiguous()

class SequentialGptOssExpert(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, x) -> torch.Tensor:
        up = self.up_proj(x)
        up = up.clamp(min=-self.limit, max=self.limit)
        gate = self.gate_proj(x)
        gate = gate.clamp(min=None, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        out = self.down_proj(gated_output)
        return out

def replace(config: GptOssConfig, module: GptOssMLP):
    return SequentialGptOssMLP(config=config, original=module)