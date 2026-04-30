from abc import ABC
from typing import Callable
import tqdm

import torch
import torch.distributed as dist
from transformers import PreTrainedModel
from transformers.core_model_loading import WeightConverter, WeightRenaming
from transformers.modeling_utils import local_torch_dtype
from transformers.integrations.moe import _default_apply_gate
from compressed_tensors.distributed import is_distributed
from compressed_tensors.offload import offload_module, get_cache_init_kwargs

from llmcompressor.utils.dev import skip_weights_initialize

from . import context
from .helpers import (
    FusedExpertsModule,
    _get_moe_shapes,
    _is_moe_experts_converter,
    _is_moe_experts_module,
)


@torch.no_grad()
def linearize_moe_model(model: PreTrainedModel):
    _weight_conversions: list[WeightConverter | WeightRenaming] = (
        model._weight_conversions
    )

    # remove all weight loading conversions; save as linearized
    print(f"removing {[converter for converter in _weight_conversions if _is_moe_experts_converter(converter)]}")
    model._weight_conversions = [
        converter
        for converter in _weight_conversions
        if not _is_moe_experts_converter(converter)
    ]

    named_experts_modules = [
        (name, module)
        for name, module in model.named_modules()
        if _is_moe_experts_module(module)
    ]

    with local_torch_dtype(model.config.dtype, model.__class__.__name__):
        for name, module in tqdm.tqdm(named_experts_modules):
            new_moe = LinearExperts.from_experts(module)
            model.set_submodule(name, new_moe)

            if is_distributed():
                dist.barrier()

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
        moe_intermediate_size: int,
        has_bias: bool,
        _apply_gate: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_dim, moe_intermediate_size, bias=has_bias)
        self.gate_proj = torch.nn.Linear(
            hidden_dim, moe_intermediate_size, bias=has_bias
        )
        self.down_proj = torch.nn.Linear(
            moe_intermediate_size, hidden_dim, bias=has_bias
        )
        self._apply_gate = _apply_gate

    @classmethod
    def from_experts(
        cls,
        experts: FusedExpertsModule,
        expert_index: int,
        moe_intermediate_size: int,
        hidden_dim: int,
    ):
        assert experts.has_gate
        # if experts.__class__._apply_gate is not _default_apply_gate:
        #     # assume that if a `_apply_gate` is implemented, then the weight
        #     # is not valid for quantization (for example, might be interleaved)
        #     raise NotImplementedError(
        #         f"Linearization for {experts.__class__.__name__} "
        #         "has not been implemented yet"
        #     )

        with skip_weights_initialize():
            instance = cls(
                hidden_dim, moe_intermediate_size, experts.has_bias, experts._apply_gate
            )

        for module in instance.modules():
            offload_module(module, **get_cache_init_kwargs(experts))

        # load weights
        gate_weight = experts.gate_up_proj[expert_index, :moe_intermediate_size]
        up_weight = experts.gate_up_proj[expert_index, moe_intermediate_size:]
        down_weight = experts.down_proj[expert_index]

        if experts.is_transposed:
            gate_weight = gate_weight.T
            up_weight = up_weight.T
            down_weight = down_weight.T

        instance.gate_proj.weight.copy_(gate_weight)
        instance.up_proj.weight.copy_(up_weight)
        instance.down_proj.weight.copy_(down_weight)

        # load biases
        if experts.has_bias:
            gate_bias = experts.gate_up_proj_bias[expert_index, :moe_intermediate_size]
            up_bias = experts.gate_up_proj_bias[expert_index, moe_intermediate_size:]
            down_bias = experts.down_proj_bias[expert_index]

            instance.gate_proj.bias.copy_(gate_bias)
            instance.up_proj.bias.copy_(up_bias)
            instance.down_proj.bias.copy_(down_bias)

        return instance

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        moe_intermediate_size: int,
        has_bias: bool,
        act_fn: torch.nn.Module,
    ):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_dim, moe_intermediate_size, bias=has_bias)
        self.down_proj = torch.nn.Linear(
            moe_intermediate_size, hidden_dim, bias=has_bias
        )
        self.act_fn = act_fn

    @classmethod
    def from_experts(
        cls,
        experts: FusedExpertsModule,
        expert_index: int,
        moe_intermediate_size: int,
        hidden_dim: int,
    ):
        assert not experts.has_gate
        if experts.__class__._apply_gate is not _default_apply_gate:
            raise ValueError("TODO")

        with skip_weights_initialize():
            instance = cls(
                hidden_dim, moe_intermediate_size, experts.has_bias, experts.act_fn
            )

        # load weights
        up_weight = experts.up_proj[expert_index]
        down_weight = experts.down_proj[expert_index]

        if experts.is_transposed:
            up_weight = up_weight.T
            down_weight = down_weight.T

        instance.up_proj.weight.copy_(up_weight)
        instance.down_proj.weight.copy_(down_weight)

        # load biases
        if experts.has_bias:
            up_bias = experts.up_proj_bias[expert_index]
            down_bias = experts.down_proj_bias[expert_index]

            instance.up_proj.bias.copy_(up_bias)
            instance.down_proj.bias.copy_(down_bias)

        return instance

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class LinearExperts(torch.nn.ModuleList):
    @classmethod
    def from_experts(cls, experts: FusedExpertsModule):
        num_experts, moe_intermediate_size, hidden_dim = _get_moe_shapes(experts)

        # TODO: add registry
        experts_cls = ExpertMLPWithGate if experts.has_gate else ExpertMLPWithoutGate

        return cls(
            [
                experts_cls.from_experts(
                    experts, index, moe_intermediate_size, hidden_dim
                )
                for index in range(num_experts)
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
            #if context.CALIBRATE_ALL_EXPERTS:
            # TODO: fully integrate moe context
            if True:
                expert_output = expert(hidden_states)[token_indices]
            else:
                expert_output = expert(hidden_states[token_indices])

            # apply weighting to outputs
            expert_weights = top_k_weights[token_indices, top_k_pos, None]
            weighted_output = expert_output * expert_weights

            # accumulate the selected tokens
            final_hidden_states.index_add_(0, token_indices, weighted_output.to(final_hidden_states.dtype))  # TODO: check why float

        return final_hidden_states
