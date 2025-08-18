from typing import Tuple

import torch
import transformers
from packaging import version
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextExperts,
    Llama4TextMLP,
    Llama4TextMoe,
)

from llmcompressor.utils.dev import skip_weights_initialize


class SequentialLlama4TextMoe(torch.nn.Module):
    def __init__(self, config: Llama4TextConfig, original: Llama4TextMoe):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = SequentialLlama4TextExperts(config, original.experts)
        self.router = original.router
        self.shared_expert = original.shared_expert

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.tensor]:
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

        if version.parse(transformers.__version__) >= version.parse("4.54.0"):
            return out, router_logits
        else:
            return out, router_scores


class SequentialLlama4TextExperts(torch.nn.ModuleList):
    def __init__(self, config: Llama4TextConfig, original: Llama4TextExperts):
        self.num_experts = original.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])

        intermediate_size = original.down_proj.shape[1]

        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]
            down = original.down_proj[i]

            gate_proj = gate_up[:, :intermediate_size]
            up_proj = gate_up[:, intermediate_size:]

            self[i].gate_proj.weight.data = gate_proj.t().clone().contiguous()
            self[i].up_proj.weight.data = up_proj.t().clone().contiguous()
            self[i].down_proj.weight.data = down.t().clone().contiguous()


def replace(config: Llama4Config, module: Llama4TextMoe):
    return SequentialLlama4TextMoe(config=config.get_text_config(), original=module)
