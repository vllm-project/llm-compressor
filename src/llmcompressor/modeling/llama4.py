from typing import Tuple

import torch
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextExperts,
    Llama4TextMLP,
    Llama4TextMoe,
)

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize


@MoECalibrationModule.register("Llama4TextMoe")
class SequentialLlama4TextMoe(MoECalibrationModule):
    """
    Calibration version of Llama4TextMoe that unpacks experts for sequential processing.

    This module:
    1. Unpacks the packed expert weights (3D -> 2D) for calibration
    2. Optionally sends all tokens to all experts during calibration
    3. Stays in unpacked form (permanent) for vLLM compatibility
    """

    is_permanent = True

    def __init__(
        self,
        original: Llama4TextMoe,
        config: Llama4Config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        # Extract text config from multimodal config
        text_config: Llama4TextConfig = config.get_text_config()
        self.top_k = text_config.num_experts_per_tok
        self.hidden_dim = text_config.hidden_size
        self.num_experts = text_config.num_local_experts

        self.experts = SequentialLlama4TextExperts(text_config, original.experts)
        self.router = original.router
        self.shared_expert = original.shared_expert
        self.calibrate_all_experts = calibrate_all_experts
        print(self.calibrate_all_experts, "CALIBRATE ALL EXPERTS")

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        # support transformers 4.53 and greater
        if isinstance(router_logits, tuple):
            router_logits = router_logits[-1]

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

        router_scores = (
            torch.full_like(router_logits, float("-inf"))
            .scatter_(1, router_indices, router_top_value)
            .transpose(0, 1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        out = self.shared_expert(hidden_states)
        for i in range(self.num_experts):
            out += self.experts[i](hidden_states) * router_scores[i].reshape(-1, 1)

        return out, router_logits


class SequentialLlama4TextExperts(torch.nn.ModuleList):
    def __init__(self, config: Llama4TextConfig, original: Llama4TextExperts):
        self.num_experts = original.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])

        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]
            down = original.down_proj[i]

            gate_proj, up_proj = gate_up.chunk(2, dim=-1)

            self[i].gate_proj.weight.data = gate_proj.t().contiguous()
            self[i].up_proj.weight.data = up_proj.t().contiguous()
            self[i].down_proj.weight.data = down.t().contiguous()
