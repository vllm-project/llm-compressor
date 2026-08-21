import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from transformers.activations import ACT2FN
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import Llama4TextExperts

from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import ExpertMLPWithGate, LinearExperts2D
from llmcompressor.utils.dev import skip_weights_initialize


class Llama4LinearExperts(LinearExperts2D):
    is_concatenated = False
    is_transposed = True
    has_bias = False
    has_gate = True

    # During all-expert calibration, every expert receives every routed token so
    # activation statistics are representative. Chunk those forwards to prevent a
    # single expert projection from materializing a multi-GiB activation tensor.
    _calibration_chunk_size = 8192

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Apply gated activation: act_fn(gate) * up"""
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    @classmethod
    @torch.no_grad()
    def from_experts_module(cls, experts: "Llama4TextExperts", config: Llama4Config):
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

        output = hidden_states.new_empty(hidden_states.shape)
        for i in range(self.num_experts):
            expert = self[i]
            if get_calibrate_all_experts_flag():
                start = i * num_tokens
                end = start + num_tokens
                for chunk_start in range(
                    0, hidden_states.shape[0], self._calibration_chunk_size
                ):
                    chunk_end = min(
                        chunk_start + self._calibration_chunk_size,
                        hidden_states.shape[0],
                    )
                    overlap_start = max(start, chunk_start)
                    overlap_end = min(end, chunk_end)
                    expert_output = expert(hidden_states[chunk_start:chunk_end])
                    if overlap_start < overlap_end:
                        output[overlap_start:overlap_end] = expert_output[
                            overlap_start - chunk_start : overlap_end - chunk_start
                        ]
            else:
                output[start:end] = expert(hidden_states[start:end])

        return output


# register in registry
LinearExperts2D._registry[Llama4TextExperts] = Llama4LinearExperts
