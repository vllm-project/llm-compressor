# flake8: noqa
from typing import List

import torch
from compressed_tensors.utils import align_module_device, update_offload_parameter
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from llmcompressor.utils.dev import skip_weights_initialize


class GptOssExpert(torch.nn.Module):
    gate_proj: torch.nn.Linear
    up_proj: torch.nn.Linear
    down_proj: torch.nn.Linear

    def __init__(self, experts: GptOssExperts):
        super().__init__()

        self.hidden_size = experts.hidden_size
        self.expert_dim = experts.expert_dim
        self.alpha = experts.alpha
        self.limit = experts.limit

        assert experts.gate_up_proj.dtype == experts.gate_up_proj_bias.dtype
        assert experts.down_proj.dtype == experts.down_proj_bias.dtype

        with skip_weights_initialize():
            self.gate_proj = torch.nn.Linear(
                self.hidden_size,
                self.expert_dim,
                bias=True,
                dtype=experts.gate_up_proj.dtype,
            )
            self.up_proj = torch.nn.Linear(
                self.hidden_size,
                self.expert_dim,
                bias=True,
                dtype=experts.gate_up_proj.dtype,
            )
            self.down_proj = torch.nn.Linear(
                self.expert_dim,
                self.hidden_size,
                bias=True,
                dtype=experts.down_proj.dtype,
            )

    def forward(self, hidden_states: torch.Tensor):
        gate = self.gate_proj(hidden_states)
        gate = gate.clamp(min=None, max=self.limit)

        up = self.up_proj(hidden_states)
        up = up.clamp(min=-self.limit, max=self.limit)

        glu = gate * torch.sigmoid(gate * self.alpha)
        return self.down_proj((up + 1) * glu)


class GptOssExpertsLinear(torch.nn.Module):
    experts: List[GptOssExpert]

    def __init__(self, experts: GptOssExperts):
        super().__init__()

        self.intermediate_size = experts.intermediate_size
        self.num_experts = experts.num_experts
        self.hidden_size = experts.hidden_size
        self.expert_dim = experts.expert_dim

        with skip_weights_initialize():
            self.experts = torch.nn.ModuleList(
                [GptOssExpert(experts) for _ in range(self.num_experts)]
            )

        self.load_weights(experts)

        self.alpha = experts.alpha
        self.limit = experts.limit

    def load_weights(self, experts: GptOssExperts):
        with align_module_device(experts):
            for expert_index, expert in enumerate(self.experts):
                update_offload_parameter(
                    expert.gate_proj,
                    "weight",
                    experts.gate_up_proj[expert_index, ..., ::2].T,
                )
                update_offload_parameter(
                    expert.gate_proj,
                    "bias",
                    experts.gate_up_proj_bias[expert_index, ..., ::2],
                )

                update_offload_parameter(
                    expert.up_proj,
                    "weight",
                    experts.gate_up_proj[expert_index, ..., 1::2].T,
                )
                update_offload_parameter(
                    expert.up_proj,
                    "bias",
                    experts.gate_up_proj_bias[expert_index, ..., 1::2],
                )

                update_offload_parameter(
                    expert.down_proj, "weight", experts.down_proj[expert_index].T
                )
                update_offload_parameter(
                    expert.down_proj, "bias", experts.down_proj_bias[expert_index]
                )

    def to_original(self) -> GptOssExperts:
        # TODO: this doesn't really handle offloading or correct device placement
        with skip_weights_initialize(use_zeros=True):
            fake_config = GptOssConfig(
                intermediate_size=self.intermediate_size,
                num_local_experts=self.num_experts,
                hidden_size=self.hidden_size,
            )
            experts = GptOssExperts(fake_config)
            experts.gate_up_proj = torch.nn.Parameter(
                experts.gate_up_proj.to(dtype=self.experts[0].gate_proj.weight.dtype),
                requires_grad=False,
            )
            experts.gate_up_proj_bias = torch.nn.Parameter(
                experts.gate_up_proj_bias.to(
                    dtype=self.experts[0].gate_proj.weight.dtype
                ),
                requires_grad=False,
            )
            experts.down_proj = torch.nn.Parameter(
                experts.down_proj.to(dtype=self.experts[0].down_proj.weight.dtype),
                requires_grad=False,
            )
            experts.down_proj_bias = torch.nn.Parameter(
                experts.down_proj_bias.to(dtype=self.experts[0].down_proj.weight.dtype),
                requires_grad=False,
            )

        for expert_index, expert in enumerate(self.experts):
            with align_module_device(expert.gate_proj, "cpu"), align_module_device(
                expert.up_proj, "cpu"
            ), align_module_device(expert.down_proj, "cpu"):
                experts.gate_up_proj[expert_index, ..., ::2].copy_(
                    expert.gate_proj.weight.data.T
                )
                experts.gate_up_proj_bias[expert_index, ..., ::2].copy_(
                    expert.gate_proj.bias.data
                )

                experts.gate_up_proj[expert_index, ..., 1::2].copy_(
                    expert.up_proj.weight.data.T
                )
                experts.gate_up_proj_bias[expert_index, ..., 1::2].copy_(
                    expert.up_proj.bias.data
                )

                experts.down_proj[expert_index].copy_(expert.down_proj.weight.data.T)
                experts.down_proj_bias[expert_index].copy_(expert.down_proj.bias.data)

                # TODO: convert qparams as well

        print("converted, for some reason slows down over time")
        import time

        print(time.time())

        experts.eval()
        return experts

    def forward(
        self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
    ) -> torch.Tensor:
        """
        When training is is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
        Returns:
            torch.Tensor
        """
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(
            -1, self.hidden_size
        )  # (num_tokens, hidden_size)

        next_states = torch.zeros_like(
            hidden_states, dtype=hidden_states.dtype, device=hidden_states.device
        )
        for expert_index, expert in enumerate(self.experts):
            next_states += expert(hidden_states) * routing_weights.T[
                expert_index
            ].unsqueeze(-1)

        next_states = next_states.reshape(original_shape)
        return next_states


def replace_gpt_oss(config: GptOssConfig, module: GptOssExpert):
    return GptOssExpertsLinear(module)


def test_restore():
    config = GptOssConfig(hidden_size=7, num_local_experts=3, expert_dim=5)

    original = GptOssExperts(config)
    linear = GptOssExpertsLinear(original)

    restored = linear.to_original()
    for param_name, param in original.named_parameters(recurse=False):
        restored_param = getattr(restored, param_name)
        assert param.shape == restored_param.shape
        assert param.dtype == restored_param.dtype

        assert torch.all(getattr(restored, param_name) == param)


def test_correctness():
    batch_size, seq_len = 13, 12
    config = GptOssConfig(hidden_size=7, num_local_experts=3, expert_dim=5)

    input = torch.rand((batch_size, seq_len, config.hidden_size))
    routing_weights = torch.rand((batch_size * seq_len, config.num_local_experts))

    with torch.no_grad():
        original = GptOssExperts(config)
        for name in [
            "gate_up_proj",
            "gate_up_proj_bias",
            "down_proj",
            "down_proj_bias",
        ]:
            setattr(original, name, getattr(original, name).normal_())

        original.eval()
        assert not original.training
        true_output = original(input, routing_weights=routing_weights)

        linear = GptOssExpertsLinear(original)
        output = linear(input, routing_weights=routing_weights)

        assert torch.allclose(output, true_output, atol=1e-3, rtol=0.0)

        restored = linear.to_original()
        restored_output = restored(input, routing_weights=routing_weights)
        assert torch.allclose(restored_output, true_output, atol=1e-3, rtol=0.0)


if __name__ == "__main__":
    test_restore()
