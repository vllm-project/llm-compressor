from typing import Callable

import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module, disable_onloading
from transformers.activations import ACT2FN
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from weakref import ref, ReferenceType
import math
from transformers.models.inkling.modeling_inkling import InklingExperts
from transformers import PreTrainedConfig
from transformers.models.inkling.configuration_inkling import InklingConfig

from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.helpers import MoEConfig
from llmcompressor.modeling.moe.linear_experts import ExpertMLP, ExpertMLPWithGate, LinearExperts2D
from llmcompressor.utils.dev import skip_weights_initialize

from collections import UserDict

class LazyDict(UserDict):
    def __init__(
        self,
        experts_ref: ReferenceType["InklingLinearExperts"],
        param_name: str,
        slice: tuple[int, slice],
    ):
        super().__init__()
        self.experts_ref = experts_ref
        self.param_name = param_name
        self.slice = slice

    def __getitem__(self, key):
        if key == "weight":
            return getattr(self.experts_ref(), self.param_name)[self.slice]
            
        return super().__getitem__(key)


class FakeLinear(torch.nn.Linear):
    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        experts_ref: ReferenceType["InklingLinearExperts"],
        slice: tuple[int, slice],
    ) -> None:
        super(torch.nn.Module, self).__init__()
        self._parameters = LazyDict(experts_ref, slice)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass.
        """
        return torch.nn.functional.linear(input, self.weight, None)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return "asdf" #f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class FakeExpert(ExpertMLP):
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
        experts_ref: ReferenceType["InklingLinearExperts"],
        index: int,
    ):
        self.experts_ref = experts_ref
        self.index = index

        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_proj = FakeLinear(
            hidden_dim, intermediate_size, mlp_bias, dtype, experts_ref, "gate_up_proj", [index, slice(None, intermediate_size // 2)],
        )
        self.up_proj = FakeLinear(
            hidden_dim, intermediate_size, mlp_bias, dtype, experts_ref, "gate_up_proj", [index, slice(intermediate_size // 2, None)],
        )
        self.down_proj = FakeLinear(
            intermediate_size, hidden_dim, mlp_bias, dtype, experts_ref, "down_proj", [index, None],
        )
        self._apply_gate = _apply_gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self._apply_gate(
                torch.cat(
                    [self.gate_proj(hidden_states), self.up_proj(hidden_states)], dim=-1
                )
            )
        )


class InklingLinearExperts(LinearExperts2D):
    gate_proj: torch.nn.Parameter
    up_proj: torch.nn.Parameter

    def __init__(self, config: InklingConfig, *args, **kwargs):
        moe_config = MoEConfig.from_config(config)

        # store num_experts before appending `act_fn` to module list
        self.num_experts = moe_config.num_experts
        self.intermediate_size = moe_config.intermediate_size
        act_fn: torch.nn.Module = ACT2FN[moe_config.hidden_act]

        expert_cls = FakeExpert
        post_up_fn = self._apply_gate if self.has_gate else act_fn.forward
        self_ref = ref(self)
        super().__init__(
            [
                expert_cls(
                    moe_config.hidden_dim,
                    moe_config.intermediate_size,
                    moe_config.use_bias,
                    post_up_fn,
                    moe_config.dtype,
                    self_ref,
                    index,
                )
                for index in range(moe_config.num_experts)
            ]
        )

        #for index in range(moe_config.num_experts)

        self.act_fn = act_fn
        self.alpha = moe_config.alpha
        self.limit = moe_config.limit

    @classmethod
    @torch.no_grad()
    def from_experts_module(cls, experts: InklingExperts, config: InklingConfig):
        with torch.device("meta"):
            self = cls(config)

        # copy offloading from original
        offload_kwargs = get_cache_init_kwargs(experts)
        for module in self.modules():
            offload_module(module, **offload_kwargs)

        # shallow copy parameters
        with disable_onloading():
            self.gate_proj = experts.gate_proj
            self.up_proj = experts.up_proj

        return self


# register in registry
LinearExperts2D._registry[InklingExperts] = InklingLinearExperts
