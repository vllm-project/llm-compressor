from abc import ABC
from typing import Any, Callable, ClassVar

import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from transformers import (
    PreTrainedConfig,
)
from transformers.activations import ACT2FN
from transformers.integrations.moe import _default_apply_gate

from llmcompressor.utils.dev import skip_weights_initialize

from .context import get_calibrate_all_experts_flag
from .helpers import (
    FusedExpertsProtocol,
    MoEConfig,
    get_use_experts_implementation_args,
)


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
        intermediate_size: int,
        mlp_bias: bool,
        _apply_gate: Callable[[torch.Tensor], torch.Tensor],
        dtype: torch.dtype,
    ):
        super().__init__()
        self.up_proj = torch.nn.Linear(
            hidden_dim, intermediate_size, bias=mlp_bias, dtype=dtype
        )
        self.gate_proj = torch.nn.Linear(
            hidden_dim, intermediate_size, bias=mlp_bias, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_dim, bias=mlp_bias, dtype=dtype
        )
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
        intermediate_size: int,
        mlp_bias: bool,
        act_fn: torch.nn.Module | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        assert act_fn is not None

        self.up_proj = torch.nn.Linear(
            hidden_dim, intermediate_size, bias=mlp_bias, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_dim, bias=mlp_bias, dtype=dtype
        )
        self.act_fn = act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class LinearExperts2D(torch.nn.ModuleList):
    """

    # 1. try for mappings (efficient load)
    # 2. try for standardized moe, convert after load
    # 3. Explicit replacement (GraniteMoeLinearExperts)

    """

    is_concatenated: ClassVar[bool]
    is_transposed: ClassVar[bool]
    has_bias: ClassVar[bool]
    has_gate: ClassVar[bool]
    _apply_gate: ClassVar[Callable[[torch.Tensor], torch.Tensor]]

    num_experts: int
    intermediate_size: int

    # custom model definitions
    _registry: ClassVar[dict[type[torch.nn.Module], type["LinearExperts2D"]]] = dict()

    @classmethod
    def get_registration(
        cls, key: type[torch.nn.Module], default: Any = None
    ) -> type["LinearExperts2D"]:
        from .granitemoe import GraniteMoeLinearExperts  # noqa: F401

        return cls._registry.get(key, default)

    @classmethod
    def create_linear_experts_cls(
        cls, experts_cls: type[FusedExpertsProtocol]
    ) -> type["LinearExperts2D"]:
        if linear_experts_cls := cls.get_registration(experts_cls):
            return linear_experts_cls

        experts_cls_args = get_use_experts_implementation_args(experts_cls)
        if experts_cls_args is None:
            raise ValueError(
                "Cannot create linear experts class from a class which does not have "
                "the `use_experts_implementation` argument. "
            )

        experts_cls_args["_apply_gate"] = (
            experts_cls._apply_gate
            if experts_cls_args["has_gate"]
            else _default_apply_gate
        )

        # reuse existing classes to avoid creating excessive types
        linear_experts_cls = type("LinearExperts2D", (cls,), experts_cls_args)
        cls._registry[experts_cls] = linear_experts_cls
        return linear_experts_cls

    @classmethod
    @torch.no_grad()
    def from_experts_module(
        cls, experts: FusedExpertsProtocol, config: PreTrainedConfig
    ):
        if not cls.has_gate:
            # assume that if a `_apply_gate` is implemented, then the weight
            # is not valid for quantization (for example, might be interleaved)
            raise NotImplementedError(
                f"Linearization for {experts.__class__.__name__} "
                "has not been implemented yet"
            )

        with skip_weights_initialize():
            self = cls(config)

        # TODO: experiment with copying views rather than data
        for index in range(self.num_experts):
            expert: ExpertMLPWithGate = self[index]

            # load weights
            gate_weight = experts.gate_up_proj[index, : self.intermediate_size]
            up_weight = experts.gate_up_proj[index, self.intermediate_size :]
            down_weight = experts.down_proj[index]

            if experts.is_transposed:
                gate_weight = gate_weight.T
                up_weight = up_weight.T
                down_weight = down_weight.T

            expert.gate_proj.weight.copy_(gate_weight)
            expert.up_proj.weight.copy_(up_weight)
            expert.down_proj.weight.copy_(down_weight)

            # load biases
            if experts.has_bias:
                gate_bias = experts.gate_up_proj_bias[index, : self.intermediate_size]
                up_bias = experts.gate_up_proj_bias[index, self.intermediate_size :]
                down_bias = experts.down_proj_bias[index]

                expert.gate_proj.bias.copy_(gate_bias)
                expert.up_proj.bias.copy_(up_bias)
                expert.down_proj.bias.copy_(down_bias)

        # copy offloading from original
        offload_kwargs = get_cache_init_kwargs(experts)
        for module in self.modules():
            offload_module(module, **offload_kwargs)

        return self

    def __init__(self, config: PreTrainedConfig, *args, **kwargs):
        moe_config = MoEConfig.from_config(config)

        # store num_experts before appending `act_fn` to module list
        self.num_experts = moe_config.num_experts
        self.intermediate_size = moe_config.intermediate_size
        act_fn = ACT2FN[moe_config.hidden_act]

        if self.has_gate:
            super().__init__(
                [
                    ExpertMLPWithGate(
                        moe_config.hidden_dim,
                        moe_config.intermediate_size,
                        moe_config.use_bias,
                        self._apply_gate,
                        moe_config.dtype,
                    )
                    for _ in range(moe_config.num_experts)
                ]
            )
        else:
            super().__init__(
                [
                    ExpertMLPWithoutGate(
                        moe_config.hidden_dim,
                        moe_config.intermediate_size,
                        moe_config.use_bias,
                        act_fn,
                        moe_config.dtype,
                    )
                    for _ in range(moe_config.num_experts)
                ]
            )

        self.act_fn = act_fn
        self.limit = moe_config.limit

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        # create tokens mask
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_index in range(self.num_experts):
            # select tokens for this expert
            top_k_pos, token_indices = torch.where(expert_mask[expert_index])

            # apply expert
            expert = self[expert_index]
            if get_calibrate_all_experts_flag():
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
