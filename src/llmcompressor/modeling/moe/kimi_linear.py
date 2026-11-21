"""
Linearized experts module for Kimi-K3 (kimi_linear) MoE layers.

Kimi-K3's native experts are per-expert ``KimiBlockSparseMLP`` modules with
``w1``/``w2``/``w3`` projections and a (possibly latent) MoE hidden size which
differs from the model hidden size (``routed_expert_hidden_size``). The class
here is the :class:`~llmcompressor.modeling.moe.linear_experts.LinearExperts2D`
replacement used during calibration: it constructs ``ExpertMLPWithGate`` experts
at the latent MoE width and supports the ``situ`` activation, which is not in
``ACT2FN``.

``KimiLinearExperts`` (in
``llmcompressor.modeling.kimi_k3.modeling_kimi_linear``) is the
standard-format class registered against this replacement. It is also the
fused class which ``LinearExperts2D.to_experts_module`` rebuilds when packing
a (possibly pruned) checkpoint back into the native format.
"""

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

__all__ = ["KimiLinearLinearExperts"]


class KimiLinearLinearExperts(LinearExperts2D):
    is_concatenated = False
    is_transposed = False
    has_bias = False
    has_gate = True

    def __init__(self, config: PreTrainedConfig, *args, **kwargs):
        moe_config = MoEConfig.from_config(config)
        self.num_experts = moe_config.num_experts
        self.intermediate_size = moe_config.intermediate_size

        # support initialization of situ activation, which is not in ACT2FN
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
        self.alpha = moe_config.alpha
        self.limit = moe_config.limit
        self.swiglu_limit = moe_config.limit

        # MoNe novices
        self.register_parameter("novices", None)

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        # support applying situ activation
        if isinstance(self.act_fn, SituAndMul):
            return self.act_fn(gate_up_out)
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up


# register in registry
LinearExperts2D._registry[KimiLinearExperts] = KimiLinearLinearExperts
