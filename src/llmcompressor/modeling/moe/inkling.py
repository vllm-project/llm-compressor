from typing import Callable, ClassVar

import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module, disable_onloading
from transformers.integrations.moe import _default_apply_gate
from weakref import ref, ReferenceType
import math
from transformers.models.inkling.modeling_inkling import InklingExperts
from transformers.models.inkling.configuration_inkling import InklingConfig

from llmcompressor.modeling.moe.linear_experts import ExpertMLP, ExpertMLPWithGate, LinearExperts2D

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

    def __contains__(self, key):
        return key == "weight"

    def __getitem__(self, key):
        if key == "weight":
            return getattr(self.experts_ref(), self.param_name)[self.slice]
            
        return super().__getitem__(key)


class InklingLinearExperts(LinearExperts2D):
    is_concatenated = False
    is_transposed = False
    has_bias = False
    has_gate = True
    _apply_gate = _default_apply_gate

    gate_up_proj: torch.nn.Parameter
    down_proj: torch.nn.Parameter

    def __init__(self, config: InklingConfig, *args, **kwargs):
        super().__init__(config)

        for index in range(self.num_experts):
            expert: ExpertMLP = self[index]
            expert.gate_proj._parameters = LazyDict(ref(self), "gate_up_proj", [index, slice(None, self.intermediate_size)])
            expert.gate_proj._buffers = {"bias": None}
            expert.up_proj._parameters = LazyDict(ref(self), "gate_up_proj", [index, slice(self.intermediate_size, None)])
            expert.up_proj._buffers = {"bias": None}
            expert.down_proj._parameters = LazyDict(ref(self), "down_proj", [index, slice(None)])
            expert.down_proj._buffers = {"bias": None}

    @classmethod
    @torch.no_grad()
    def from_experts_module(cls, experts: InklingExperts, config: InklingConfig):
        with torch.device("meta"):
            self = cls(config)

        # # copy offloading from original
        # offload_kwargs = get_cache_init_kwargs(experts)
        # for module in self.modules():
        #     offload_module(module, **offload_kwargs)

        # for index in range(self.num_experts):
        #     expert: ExpertMLP = self[index]
        #     expert.gate_proj._parameters = LazyDict(ref(self), "gate_up_proj", [index, slice(None, self.intermediate_size)])
        #     expert.gate_proj._buffers = {"bias": None}
        #     expert.up_proj._parameters = LazyDict(ref(self), "gate_up_proj", [index, slice(self.intermediate_size, None)])
        #     expert.up_proj._buffers = {"bias": None}
        #     expert.down_proj._parameters = LazyDict(ref(self), "down_proj", [index, slice(None)])
        #     expert.down_proj._buffers = {"bias": None}

        # shallow copy parameters
        with disable_onloading():
            self.gate_up_proj = experts.gate_up_proj
            self.down_proj = experts.down_proj

        return self


# register in registry
LinearExperts2D._registry[InklingExperts] = InklingLinearExperts
