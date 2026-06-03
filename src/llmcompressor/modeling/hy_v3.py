from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers.models.hy_v3.modeling_hy_v3 import HYV3MoE


@MoECalibrationModule.register("HYV3MoE")
class CalibrationHYV3MoE(MoECalibrationModule):
    """
    Calibration version of HYV3MoE that unfuses expert parameters into
    individual MLP modules so that expert weights can be targeted as Linear
    layers. Sends all tokens to all experts during calibration while preserving
    the original routed output.

    is_permanent = True because the unfused structure must persist through
    finalization for quantization to target the individual expert Linears.
    """

    is_permanent = True

    def __init__(
        self,
        original: HYV3MoE,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        text_config = getattr(config, "text_config", config)

        self.calibrate_all_experts = calibrate_all_experts
        self.top_k = original.top_k
        self.num_experts = original.experts.num_experts
        self.hidden_dim = original.experts.hidden_dim
        self.hidden_size = original.experts.hidden_dim
        self.enable_moe_fp32_combine = original.enable_moe_fp32_combine

        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.register_buffer(
            "e_score_correction_bias",
            original.e_score_correction_bias,
            persistent=True,
        )
        self.experts = SequentialHYV3Experts(text_config, original.experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        _, top_k_weights, top_k_index = self.gate(
            hidden_states_reshaped, self.e_score_correction_bias
        )

        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(
            2, 1, 0
        )
        routed_output = torch.zeros_like(hidden_states_reshaped)

        for expert_idx, expert_layer in enumerate(self.experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states_reshaped)[token_idx]
            elif len(token_idx) > 0:
                expert_out = expert_layer(hidden_states_reshaped[token_idx])
            else:
                continue

            if len(token_idx) > 0:
                current_hidden_states = (
                    expert_out * top_k_weights[token_idx, top_k_pos, None]
                )
                routed_output.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(routed_output.dtype),
                )

        shared_output = self.shared_experts(hidden_states_reshaped)
        if self.enable_moe_fp32_combine:
            hidden_states = (routed_output.float() + shared_output.float()).to(
                hidden_states_reshaped.dtype
            )
        else:
            hidden_states = routed_output + shared_output

        return hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return self


class SequentialHYV3Experts(torch.nn.ModuleList):
    """
    Unfuses HYV3Experts 3D tensors into individual HYV3MLP modules so that
    gate, up, and down projections are regular nn.Linear modules.
    """

    def __init__(self, config, original):
        from transformers.models.hy_v3.modeling_hy_v3 import HYV3MLP

        self.num_experts = original.num_experts
        intermediate_size = original.intermediate_dim

        with skip_weights_initialize():
            super().__init__(
                [
                    HYV3MLP(config, intermediate_size=intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        gate_up_data = original.gate_up_proj.data
        down_data = original.down_proj.data

        for expert_idx in range(self.num_experts):
            gate_up = gate_up_data[expert_idx]
            down = down_data[expert_idx]

            _remove_optional_bias(self[expert_idx].gate_proj)
            _remove_optional_bias(self[expert_idx].up_proj)
            _remove_optional_bias(self[expert_idx].down_proj)
            self[expert_idx].gate_proj.weight = _as_parameter(
                gate_up[:intermediate_size, :],
                self[expert_idx].gate_proj.weight.requires_grad,
            )
            self[expert_idx].up_proj.weight = _as_parameter(
                gate_up[intermediate_size:, :],
                self[expert_idx].up_proj.weight.requires_grad,
            )
            self[expert_idx].down_proj.weight = _as_parameter(
                down,
                self[expert_idx].down_proj.weight.requires_grad,
            )


def _as_parameter(tensor: torch.Tensor, requires_grad: bool) -> torch.nn.Parameter:
    return torch.nn.Parameter(tensor.clone().contiguous(), requires_grad=requires_grad)


def _remove_optional_bias(linear: torch.nn.Linear) -> None:
    if linear.bias is not None:
        linear.bias = None
