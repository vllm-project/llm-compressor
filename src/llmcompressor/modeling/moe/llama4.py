import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from transformers.activations import ACT2FN
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import Llama4TextExperts

from llmcompressor.modeling.moe.linear_experts import ExpertMLPWithGate, LinearExperts2D
from llmcompressor.utils.dev import skip_weights_initialize


class Llama4LinearExperts(LinearExperts2D):
    is_concatenated = False
    is_transposed = True
    has_bias = False
    has_gate = True

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Apply gated activation: act_fn(gate) * up"""
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    @classmethod
    @torch.no_grad()
    def from_experts_module(
        cls,
        experts: "Llama4TextExperts",
        config: Llama4Config,
        calibrate_all_experts: bool = True,
    ):
        config: Llama4TextConfig = config.text_config
        assert experts.num_experts == config.num_local_experts
        experts.is_concatenated = cls.is_concatenated
        experts.is_transposed = cls.is_transposed
        experts.has_bias = cls.has_bias
        experts.has_gate = cls.has_gate

        with skip_weights_initialize():
            self = cls(
                experts.num_experts,
                experts.hidden_size,
                experts.expert_dim,
                config,
            )
            self.num_experts = experts.num_experts
            self.calibrate_all_experts = calibrate_all_experts

        # Extract individual expert weights from the batched parameters
        for index in range(experts.num_experts):
            expert: ExpertMLPWithGate = self[index]
            expert.copy_from_experts_module(experts, index)

        # copy offloading from original
        offload_kwargs = get_cache_init_kwargs(experts)
        for module in self.modules():
            offload_module(module, **offload_kwargs)

        return self

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        expert_dim: int,
        config: Llama4TextConfig,
    ) -> None:
        self.num_experts = num_experts
        self.input_size = hidden_size
        self.output_size = hidden_size
        self.intermediate_size = expert_dim

        # Create expert modules with gate_proj, up_proj, and down_proj
        torch.nn.ModuleList.__init__(
            self,
            [
                ExpertMLPWithGate(
                    hidden_dim=hidden_size,
                    intermediate_size=expert_dim,
                    mlp_bias=False,
                    _apply_gate=self._apply_gate,
                    dtype=config.dtype,
                )
                for _ in range(num_experts)
            ],
        )

        # Set activation function
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching Llama4TextExperts behavior.

        Args:
            hidden_states (Tensor): (num_experts * num_tokens_per_expert, hidden_size)
                Expected to be pre-sorted by expert.

        Returns:
            Tensor: (num_experts * num_tokens_per_expert, hidden_size)
        """
        num_tokens = hidden_states.shape[0] // self.num_experts
        expert_view = (self.num_experts, num_tokens, self.input_size)

        output_list = []
        for i in range(self.num_experts):
            expert = self[i]
            if self.calibrate_all_experts:
                expert_output = expert(hidden_states).view(*expert_view)[i]
            else:
                expert_output = expert(hidden_states.view(*expert_view)[i])
            output_list.append(expert_output)

        return torch.cat(output_list, dim=0)


# register in registry
LinearExperts2D._registry[Llama4TextExperts] = Llama4LinearExperts
