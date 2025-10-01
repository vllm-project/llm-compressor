# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class Qwen3NextSparseMoeBlock(torch.nn.Module):
    def __init__(
        self,
        config,
        original,
        calibrate_all_experts: bool,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate
        self.experts = original.experts

        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be
        # sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the
        # computation on each expert
        # expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states)[top_x]
            else:
                expert_out = expert_layer(hidden_states[top_x])

            # Index the correct hidden states and compute the expert hidden
            # state for the current expert. We need to make sure to multiply
            # the output hidden states by `routing_weights` on the
            # corresponding tokens (top-1 and top-2)
            if len(top_x) > 0:
                current_hidden_states = expert_out * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(
                    0,
                    top_x,
                    current_hidden_states.to(hidden_states.dtype),
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
        return final_hidden_states, router_logits


def replace(
    config,
    module,
    calibrate_all_experts,
):
    return Qwen3NextSparseMoeBlock(
        config=config, original=module, calibrate_all_experts=calibrate_all_experts
    )
