from abc import ABC
from typing import Callable

import torch
from transformers import (
    PreTrainedConfig,
)
from transformers.activations import ACT2FN
from transformers.integrations.moe import _default_apply_gate

from .context import get_moe_calibration_context
from .helpers import (
    get_moe_dims,
    get_use_experts_implementation_args,
)


# probably only need to registry this class
class ExpertMLP(torch.nn.Module, ABC):
    pass


class ExpertMLPWithGate(ExpertMLP):
    up_proj: torch.nn.Linear
    gate_proj: torch.nn.Linear
    down_proj: torch.nn.Linear
    _apply_gate: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        mlp_bias: bool,
        _apply_gate: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=mlp_bias)
        self.gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=mlp_bias)
        self.down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, bias=mlp_bias)
        self._apply_gate = _apply_gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # TODO: handle is_transposed
        return self.down_proj(
            self._apply_gate(
                torch.cat(
                    [self.gate_proj(hidden_states), self.up_proj(hidden_states)], dim=-1
                )
            )
        )


class ExpertMLPWithoutGate(ExpertMLP):
    up_proj: torch.nn.Linear
    down_proj: torch.nn.Linear
    act_fn: torch.nn.Module

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        mlp_bias: bool,
        act_fn: torch.nn.Module | None,
    ):
        super().__init__()
        assert act_fn is not None

        self.up_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=mlp_bias)
        self.down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, bias=mlp_bias)
        self.act_fn = act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class LinearExperts2D(torch.nn.ModuleList):
    is_concatenated: bool
    is_transposed: bool
    has_bias: bool
    has_gate: bool
    _apply_gate: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, config: PreTrainedConfig, *args, **kwargs):
        num_experts, hidden_dim, intermediate_dim, mlp_bias, hidden_act, limit = (
            get_moe_dims(config)
        )

        # store num_experts before appending `act_fn` to module list
        self.num_experts = num_experts
        act_fn = ACT2FN[hidden_act]

        if self.has_gate:
            super().__init__(
                [
                    ExpertMLPWithGate(
                        hidden_dim, intermediate_dim, mlp_bias, self._apply_gate
                    )
                    for _ in range(num_experts)
                ]
            )
        else:
            super().__init__(
                [
                    ExpertMLPWithoutGate(hidden_dim, intermediate_dim, mlp_bias, act_fn)
                    for _ in range(num_experts)
                ]
            )

        self.act_fn = act_fn
        self.limit = limit

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Match the eager implementation from transformers exactly
        final_hidden_states = torch.zeros_like(hidden_states)

        # create tokens mask
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            # select tokens for this expert
            top_k_pos, token_indices = torch.where(expert_mask[expert_idx])

            # apply expert
            expert = self[expert_idx]
            if get_moe_calibration_context():
                expert_output = expert(hidden_states)[token_indices]
            else:
                expert_output = expert(hidden_states[token_indices])

            # apply weighting to outputs
            expert_weights = top_k_weights[token_indices, top_k_pos, None]
            weighted_output = expert_output * expert_weights

            # accumulate using index_add_ to match eager implementation exactly
            final_hidden_states.index_add_(
                0, token_indices, weighted_output.to(final_hidden_states.dtype)
            )

        return final_hidden_states


def create_linear_experts_2d(experts_cls: type) -> type[LinearExperts2D]:
    experts_cls_args = get_use_experts_implementation_args(experts_cls)
    # TODO: parameterize on whether the checkpoint has gate_up concatted

    if experts_cls_args["has_gate"]:
        experts_cls_args["_apply_gate"] = experts_cls._apply_gate
    else:
        experts_cls_args["_apply_gate"] = _default_apply_gate

    return type("LinearExperts2D", (LinearExperts2D,), experts_cls_args)
