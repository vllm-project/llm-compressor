import torch
from transformers import PreTrainedConfig
from transformers.activations import ACT2FN

from llmcompressor.modeling.kimi_k3.modeling_kimi_linear import (
    KimiLinearExperts,
    SituAndMul,
    _get_situ_activation_params,
)
from llmcompressor.modeling.moe.helpers import MoEConfig
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D


class KimiLinearLinearExperts(LinearExperts2D):
    is_concatenated = False
    is_transposed = False
    has_bias = False
    has_gate = True

    def __init__(self, config: PreTrainedConfig, *args, **kwargs):
        moe_config = MoEConfig.from_config(config)
        self.num_experts = moe_config.num_experts
        self.intermediate_size = moe_config.intermediate_size

        # support initialization of situ activation
        if moe_config.hidden_act == "situ":
            beta, linear_beta = _get_situ_activation_params(config)
            act_fn = SituAndMul(
                beta=beta,
                linear_beta=linear_beta,
            )
        else:
            act_fn: torch.nn.Module = ACT2FN[moe_config.hidden_act]

        torch.nn.ModuleList.__init__(
            self,
            [
                self.expert_cls_with_gate(
                    moe_config.hidden_dim,
                    moe_config.intermediate_size,
                    moe_config.use_bias,
                    self._apply_gate,
                    moe_config.dtype,
                )
                for _ in range(moe_config.num_experts)
            ],
        )

        self.act_fn = act_fn

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        # support applying situ activation
        if isinstance(self.act_fn, SituAndMul):
            return self.act_fn(gate_up_out)
        else:
            gate, up = gate_up_out.chunk(2, dim=-1)
            return self.act_fn(gate) * up


# register in registry
# in practice, KimiLinearExperts is never actually initialized
# since LLM Compressor owns the definition directly
LinearExperts2D._registry[KimiLinearExperts] = KimiLinearLinearExperts
