import torch
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize


@MoECalibrationModule.register("Qwen3_5MoeSparseMoeBlock")
class CalibrateQwen3_5MoeTextSparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of Qwen3_5MoeSparseMoeBlock that sends all tokens to all
    experts.
    """

    is_permanent = True

    def __init__(
        self,
        original: "Qwen3_5MoeSparseMoeBlock",
        config: "Qwen3_5MoeConfig",
        calibrate_all_experts: bool,
    ):
        super().__init__()
        text_config: "Qwen3_5MoeTextConfig" = config.get_text_config()

        self.num_experts = text_config.num_experts

        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.gate = original.gate
        self.experts = SequentialQwen3VLMoeTextExperts(text_config, original.experts)

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original


class SequentialQwen3VLMoeTextExperts(torch.nn.ModuleList):
    def __init__(self, config, original):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeMLP,
        )

        self.num_experts = original.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__(
                [
                    Qwen3_5MoeMLP(
                        config, intermediate_size=config.shared_expert_intermediate_size
                    )
                    for _ in range(self.num_experts)
                ]
            )

        intermediate_size = original.down_proj.shape[1]

        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]
            down = original.down_proj[i]

            gate_proj = gate_up[:, :intermediate_size]
            up_proj = gate_up[:, intermediate_size:]

            self[i].gate_proj.weight.data = gate_proj.t().clone().contiguous()
            self[i].up_proj.weight.data = up_proj.t().clone().contiguous()
            self[i].down_proj.weight.data = down.t().clone().contiguous()
