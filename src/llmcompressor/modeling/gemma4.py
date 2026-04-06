from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4Config,
        Gemma4TextExperts,
    )


@MoECalibrationModule.register("Gemma4TextExperts")
class SequentialGemma4TextExperts(MoECalibrationModule):
    """
    Calibration version of Gemma4TextExperts that unpacks experts.

    This module unpacks the packed expert weights (3D -> 2D) for calibration and
    stays in unpacked form (permanent) for vLLM compatibility.
    """

    is_permanent = True

    def __init__(
        self,
        original: Gemma4TextExperts,
        config: Gemma4Config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.num_experts = original.num_experts
        self.hidden_dim = original.hidden_dim
        self.intermediate_dim = original.intermediate_dim
        self.calibrate_all_experts = calibrate_all_experts

        # Unpack the 3D expert weights into individual MLP modules
        # Register experts directly as numbered children to avoid double nesting
        # (HF has layers[i].experts, so we want layers[i].experts.0,
        # not layers[i].experts.experts.0)
        expert_list = Gemma4TextExpertsList(config.text_config, original)
        for i, expert in enumerate(expert_list):
            self.add_module(str(i), expert)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            expert_layer = getattr(self, str(expert_idx))

            if self.calibrate_all_experts:
                # Pass all tokens through expert, then select routed outputs
                expert_out = expert_layer(hidden_states)[token_idx]
            else:
                # Only pass routed tokens through expert
                expert_out = expert_layer(hidden_states[token_idx])

            if len(token_idx) > 0:
                current_hidden_states = (
                    expert_out * top_k_weights[token_idx, top_k_pos, None]
                )
                final_hidden_states.index_add_(
                    0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
                )

        return final_hidden_states


class Gemma4TextExpertsList(torch.nn.ModuleList):
    """
    Unpacks 3D expert parameter tensors into individual Gemma4TextMLP modules
    so that each expert's weights are nn.Linear and can be targeted by
    quantization with targets="Linear".
    """

    def __init__(self, config: Gemma4TextConfig, original: Gemma4TextExperts):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

        self.num_experts = config.num_experts
        intermediate_size = config.moe_intermediate_size

        with skip_weights_initialize():
            super().__init__(
                [Gemma4TextMLP(config, layer_idx=0) for _ in range(self.num_experts)]
            )

        gate_up_data = original.gate_up_proj.data  # [num_experts, 2*inter, hidden]
        down_data = original.down_proj.data  # [num_experts, hidden, inter]

        for i in range(self.num_experts):
            gate_up = gate_up_data[i]  # [2*intermediate, hidden]
            down = down_data[i]  # [hidden, intermediate]

            # gate_up_proj stores [gate; up] stacked along dim 0
            # nn.Linear weight is [out_features, in_features]
            self[i].gate_proj.weight.data = (
                gate_up[:intermediate_size, :].clone().contiguous()
            )
            self[i].up_proj.weight.data = (
                gate_up[intermediate_size:, :].clone().contiguous()
            )
            self[i].down_proj.weight.data = down.clone().contiguous()
