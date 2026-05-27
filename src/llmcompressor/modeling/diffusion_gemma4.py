from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers.models.diffusion_gemma4.configuration_diffusion_gemma4 import (
        DiffusionGemma4TextConfig,
    )
    from transformers.models.diffusion_gemma4.modeling_diffusion_gemma4 import (
        DiffusionGemma4Config,
        DiffusionGemma4TextExperts,
    )


@MoECalibrationModule.register("DiffusionGemma4TextExperts")
class CalibrationDiffusionGemma4TextExperts(MoECalibrationModule):
    """
    Calibration version of DiffusionGemma4TextExperts that unpacks experts.

    This module unpacks the packed expert weights (3D -> 2D) for calibration and
    stays in unpacked form (permanent) for vLLM compatibility.

    The DiffusionGemma4TextExperts stores expert weights as 3D tensors:
    - gate_up_proj: [num_experts, 2*intermediate, hidden]
    - down_proj: [num_experts, hidden, intermediate]

    This class linearizes them into individual MLP modules with nn.Linear layers
    so they can be targeted by quantization with targets="Linear".
    """

    is_permanent = True

    def __init__(
        self,
        original: DiffusionGemma4TextExperts,
        config: DiffusionGemma4Config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.num_experts = original.num_experts
        self.hidden_dim = original.hidden_dim
        self.intermediate_dim = original.intermediate_dim
        self.calibrate_all_experts = calibrate_all_experts
        self.act_fn = original.act_fn

        # Unpack the 3D expert weights into individual MLP modules
        # Register experts directly as numbered children to avoid double nesting
        expert_list = SequentialDiffusionGemma4TextExperts(
            config.text_config, original
        )
        for i, expert in enumerate(expert_list):
            self.add_module(str(i), expert)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Keep original forward logic but use linearized experts
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)

        # Loop over all experts (torch.compile friendly - no data-dependent branches)
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            # Use linearized expert instead of F.linear on 3D tensors
            expert_layer = getattr(self, str(expert_idx))

            if self.calibrate_all_experts:
                # Pass all tokens through expert, then select routed outputs
                # This ensures all experts see data during calibration
                current_hidden_states = expert_layer(hidden_states)[token_idx]
            else:
                # Only pass routed tokens through expert (standard inference)
                current_state = hidden_states[token_idx]
                current_hidden_states = expert_layer(current_state)

            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            # index_add_ handles empty indices gracefully
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


class SequentialDiffusionGemma4TextExperts(torch.nn.ModuleList):
    """
    Unpacks 3D expert parameter tensors into individual DiffusionGemmaText4MLP modules
    so that each expert's weights are nn.Linear and can be targeted by
    quantization with targets="Linear".
    """

    def __init__(
        self, config: DiffusionGemma4TextConfig, original: DiffusionGemma4TextExperts
    ):
        from transformers.models.diffusion_gemma4.modeling_diffusion_gemma4 import (
            DiffusionGemmaText4MLP,
        )

        self.num_experts = config.num_experts
        intermediate_size = config.moe_intermediate_size

        with skip_weights_initialize():
            super().__init__(
                [
                    DiffusionGemmaText4MLP(config, layer_idx=0)
                    for _ in range(self.num_experts)
                ]
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
