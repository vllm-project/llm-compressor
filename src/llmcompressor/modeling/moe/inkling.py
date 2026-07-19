import torch
from compressed_tensors.offload import (
    disable_onloading,
    get_cache_init_kwargs,
    offload_module,
)
from transformers.integrations.moe import _default_apply_gate
from transformers.models.inkling.configuration_inkling import InklingConfig
from transformers.models.inkling.modeling_inkling import InklingExperts

from llmcompressor.modeling.moe.linear_experts import ExpertMLP, LinearExperts2D


class InklingLinearExperts(LinearExperts2D):
    is_concatenated = False
    is_transposed = False
    has_bias = False
    has_gate = True
    _apply_gate = _default_apply_gate

    gate_up_proj: torch.nn.Parameter
    down_proj: torch.nn.Parameter

    @classmethod
    @torch.no_grad()
    def from_experts_module(cls, experts: InklingExperts, config: InklingConfig):
        with torch.device("meta"):
            self = cls(config)

        # zero copy: remove weight to avoid offloading
        for index in range(self.num_experts):
            expert: ExpertMLP = self[index]
            del expert.gate_proj.weight
            del expert.up_proj.weight
            del expert.down_proj.weight

        # copy offloading from original
        offload_kwargs = get_cache_init_kwargs(experts)
        for module in self.modules():
            offload_module(module, **offload_kwargs)

        # zero copy: assign linear weights as a view
        with disable_onloading():
            for index in range(self.num_experts):
                expert: ExpertMLP = self[index]
                expert.gate_proj._parameters["weight"] = experts.gate_up_proj[
                    index, : self.intermediate_size
                ]
                expert.up_proj._parameters["weight"] = experts.gate_up_proj[
                    index, self.intermediate_size :
                ]
                expert.down_proj._parameters["weight"] = experts.down_proj[index]

        return self


# register in registry
LinearExperts2D._registry[InklingExperts] = InklingLinearExperts
