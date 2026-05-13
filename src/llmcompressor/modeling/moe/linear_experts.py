from abc import ABC
from typing import Callable

import torch
from transformers import (
    PreTrainedConfig,
)
from transformers.integrations.moe import _default_apply_gate
from transformers.modeling_utils import local_torch_dtype

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


class ExpertMLPWithFusedGate(ExpertMLP):
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
        self.gate_up_proj = torch.nn.Linear(
            hidden_dim, intermediate_dim * 2, bias=mlp_bias
        )
        self.down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, bias=mlp_bias)
        self._apply_gate = _apply_gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # TODO: handle is_transposed
        return self.down_proj(self._apply_gate(self.gate_up_proj(hidden_states)))


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
    """
    Args:
        experts_class (`type[torch.nn.Module]`, *optional*):
            The experts class to modify. If not provided, returns a decorator that can be applied to the class.
        experts_interface (`ExpertsInterface`, *optional*, defaults to `ALL_EXPERTS_FUNCTIONS`):
            The experts interface to use for dispatching the forward method.
        is_concatenated (`bool`, *optional*, defaults to `True`):
            Whether the expert weights are stored in concatenated layout [gate;up]
            or interleaved layout [gate0, up0, gate1, up1, ...].
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the expert weights are stored in transposed format.
        has_bias (`bool`, *optional*, defaults to `False`):
            Whether the expert layers include bias terms or not.
        has_gate (`bool`, *optional*, defaults to `True`):
            Whether the experts use a gating mechanism or not.
            Whether it has gate_up_proj weights or just up_proj weights.
    """

    is_concatenated: bool
    is_transposed: bool
    has_bias: bool
    has_gate: bool
    _apply_gate: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, config: PreTrainedConfig, *args, **kwargs):
        num_experts, hidden_dim, intermediate_dim, mlp_bias, act_fn = get_moe_dims(
            config
        )

        with local_torch_dtype(config.dtype):
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
                        ExpertMLPWithoutGate(
                            hidden_dim, intermediate_dim, mlp_bias, act_fn
                        )
                        for _ in range(num_experts)
                    ]
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        num_experts = len(self)

        # create tokens mask
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(num_experts):
            # select tokens for this expert
            top_k_pos, token_indices = torch.where(expert_mask[expert_idx])

            # apply expert, maybe pass all tokens to the expert
            expert = self[expert_idx]
            # if context.CALIBRATE_ALL_EXPERTS:
            # TODO: fully integrate moe context
            if False:
                expert_output = expert(hidden_states)[token_indices]
            else:
                expert_output = expert(hidden_states[token_indices])

            # apply weighting to outputs
            expert_weights = top_k_weights[token_indices, top_k_pos, None]
            weighted_output = expert_output * expert_weights

            # accumulate the selected tokens
            final_hidden_states.index_add_(
                0, token_indices, weighted_output.to(final_hidden_states.dtype)
            )  # TODO: check why float

        return final_hidden_states


def create_linear_experts_2d(experts_cls: type) -> type[LinearExperts2D]:
    """Factory for creating LinearExperts2D classes with decorator args from the original experts class.

    Args:
        experts_cls: The original experts class decorated with @use_experts_implementation

    Returns:
        A new LinearExperts2D class with decorator arguments as class attributes
    """
    experts_cls_args = get_use_experts_implementation_args(experts_cls)
    # TODO: parameterize on whether the checkpoint has gate_up concatted

    if experts_cls_args["has_gate"]:
        experts_cls_args["_apply_gate"] = experts_cls._apply_gate
    else:
        experts_cls_args["_apply_gate"] = _default_apply_gate

    return type("LinearExperts2D", (LinearExperts2D,), experts_cls_args)
