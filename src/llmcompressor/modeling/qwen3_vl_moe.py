import torch

from llmcompressor.utils.dev import skip_weights_initialize


class LinearQwen3VLMoeTextSparseMoeBlock(torch.nn.Module):
    def __init__(self, config, original):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.gate = wrap_gate(original.gate)
        self.experts = SequentialQwen3VLMoeTextExperts(config, original.experts)


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


def wrap_gate(gate):
    # temporary workaround until ct supports ignores of Linear instances
    linear_gate = torch.nn.Linear(gate.in_features, gate.out_features)
    linear_gate.weight.data.copy_(gate.weight.data)
    setattr(linear_gate, "hidden_size", gate.hidden_size)
    setattr(linear_gate, "top_k", gate.top_k)
    setattr(linear_gate, "forward", gate.forward)
    del gate
    return linear_gate


def replace(config, module, calibrate_all_experts=False):
    return LinearQwen3VLMoeTextSparseMoeBlock(
        config=config.get_text_config(),
        original=module,
    )
