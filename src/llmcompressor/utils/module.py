from typing import Callable, Union

import torch
import tqdm
from torch.nn import Module

from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE


def module_bfs(
    module: Module,
    func: Callable[[Module], Module],
    pre: bool = True,
    progress: Union[bool, tqdm.tqdm] = False,
) -> Module:
    if progress is True:
        total = len(list(module.modules()))
        progress = tqdm.tqdm(total=total)

    if pre:
        module = func(module)

    for name, child in list(module.named_children()):
        module.add_module(name, module_bfs(child, func, pre, progress))

    if not pre:
        module = func(module)

    if isinstance(progress, tqdm.tqdm):
        progress.update(1)

    return module

class DeepseekV3MoELinears(DeepseekV3MoE):
    def __init__(self, config, experts, gate, shared_experts):
        super(torch.nn.Module).__init__()
        self.config = config
        self.experts = experts
        self.gate = gate
        self.shared_experts = shared_experts
        
    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # self.moe
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
        # self.moe
        # hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)

        del expert_mask
        hidden_states = final_hidden_states.type(hidden_states.dtype).view(*orig_shape)

        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states