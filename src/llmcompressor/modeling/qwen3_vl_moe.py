import torch
from transformers import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock as OriginalQwen3VLMoeTextSparseMoeBlock,
)

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize


@MoECalibrationModule.register("Qwen3VLMoeTextSparseMoeBlock")
class CalibrateQwen3VLMoeTextSparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of Qwen3VLMoeTextSparseMoeBlock that sends all tokens to all
    experts.
    """

    is_permanent = True

    def __init__(
        self,
        original: OriginalQwen3VLMoeTextSparseMoeBlock,
        config: Qwen3VLMoeConfig,
        calibrate_all_experts: bool,
    ):
        super().__init__()
        text_config: Qwen3VLMoeTextConfig = config.get_text_config()

        self.hidden_size = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = original.top_k
        # Note: gate was changed to be a Linear layer in transformers==4.57.0
        # https://github.com/JJJYmmm/transformers/commit/f5dea1c694af8c994c769170813a8702332119ee
        self.gate = original.gate
        self.calibrate_all_experts = calibrate_all_experts
        self.experts = SequentialQwen3VLMoeTextExperts(text_config, original.experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        # get topk experts per token
        # routing_weight: (num_tokens, top_k)
        # routing_indices: (num_tokens, top_k)
        routing_weights, router_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        next_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # convert router indices into OHE list
        # reshape to be (num_experts, top_k, batch_size * sequence_length)
        expert_mask = torch.nn.functional.one_hot(
            router_indices, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, token_idx = torch.where(expert_mask[expert_idx].squeeze(0))

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states)[token_idx]
            else:
                expert_out = expert_layer(hidden_states[token_idx])

            if len(token_idx) > 0:
                # if there are tokens meant for this expert, further scale the expert
                # output by the score
                weighted_output = expert_out * routing_weights[token_idx, idx, None]
                next_states.index_add_(
                    0, token_idx, weighted_output.to(hidden_states.dtype)
                )

        next_states = next_states.reshape(batch_size, sequence_length, hidden_dim)
        return next_states, router_logits

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original


class SequentialQwen3VLMoeTextExperts(torch.nn.ModuleList):
    def __init__(self, config, original):
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextMLP,
        )

        self.num_experts = original.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__(
                [Qwen3VLMoeTextMLP(config) for _ in range(self.num_experts)]
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


def replace(
    config: Qwen3VLMoeConfig,
    original: OriginalQwen3VLMoeTextSparseMoeBlock,
    calibrate_all_experts: bool,
):
    return CalibrateQwen3VLMoeTextSparseMoeBlock(
        original=original,
        config=config,
        calibrate_all_experts=calibrate_all_experts,
    )
