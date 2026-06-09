"""
Calibration wrapper for Mistral 4 MoE (``transformers>=5.4``).

``Mistral4MoE`` (in ``transformers.models.mistral4.modeling_mistral4``) stores
its routed experts as fused 3D tensors and routes through a data-dependent loop
in ``Mistral4NaiveMoe.forward``::

    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        ...

The data-dependent ``expert_hit`` loop plus the ``continue`` conditional are not
capturable by ``torch.fx``, so ``SequentialPipeline`` produces a degenerate
subgraph and calibration fails with
``TypeError: unsupported operand type(s) for +: 'Tensor' and 'NoneType'``.

This mirrors the two existing patterns in this package:

* ``llama4.py`` / ``qwen3_5_moe.py`` — un-fuse 3D expert tensors into per-expert
  ``nn.Linear`` MLPs so ``targets="Linear"`` can quantize them.
* ``deepseek_v3.py`` — a static, FX-traceable expert loop with
  ``calibrate_all_experts`` semantics.

Mistral 4's MoE is a hybrid of the two (fused expert storage like Llama 4 /
Qwen3.5, but DeepSeek-V2-style group routing + a shared expert), so this wrapper
combines both: un-fuse the experts, then run a static loop that optionally feeds
all tokens to all experts for stable activation statistics.

Note: attention also carries an FX-trace blocker (``apply_rotary_pos_emb`` is
decorated with ``@use_kernel_func_from_hub``); that is handled by adding
``re:.*self_attn`` to the recipe ignore list, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
    from transformers.models.mistral4.modeling_mistral4 import (
        Mistral4MoE,
        Mistral4NaiveMoe,
    )


@MoECalibrationModule.register("Mistral4MoE")
class CalibrationMistral4MoE(MoECalibrationModule):
    """
    Calibration version of ``Mistral4MoE`` that un-fuses experts for
    FX-traceable sequential processing and (optionally) routes all tokens to all
    experts for activation calibration.

    ``is_permanent = True`` because the un-fused per-expert structure must
    persist for quantization to target the individual ``nn.Linear`` weights.
    """

    is_permanent = True

    def __init__(
        self,
        original: Mistral4MoE,
        config: Mistral4Config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()

        # Mistral 3 is the outer multimodal wrapper for Mistral 4 text; the MoE
        # config lives on the text_config sub-object. For pure-Mistral4 configs
        # ``get_text_config()`` returns self.
        text_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )

        # Routing parameters are read from the live module rather than the config
        # because the field names vary across config variants.
        self.num_experts = original.experts.num_experts
        self.top_k = original.top_k
        self.n_routed_experts = original.n_routed_experts
        self.n_group = original.n_group
        self.topk_group = original.topk_group
        self.norm_topk_prob = original.norm_topk_prob
        self.routed_scaling_factor = original.routed_scaling_factor

        # Reuse original gate + shared_experts; un-fuse the routed experts.
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.experts = SequentialMistral4Experts(text_config, original.experts)
        self.calibrate_all_experts = calibrate_all_experts

    def route_tokens_to_experts(
        self, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Group-aware top-k routing, identical to the upstream
        ``Mistral4MoE.route_tokens_to_experts`` (inlined so the calibration
        module is self-contained and the original can be released)."""
        router_logits = router_logits.softmax(-1)
        group_scores = (
            router_logits.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Fixed-bound loop is FX-traceable (unlike the upstream data-dependent
        # expert_hit loop); mirrors CalibrationDeepseekV3MoE.
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.num_experts
        ).permute(2, 0, 1)  # (num_experts, num_tokens, top_k)

        for expert_idx in range(self.num_experts):
            token_indices, weight_indices = torch.where(expert_mask[expert_idx])
            has_tokens = token_indices.numel() > 0

            if self.calibrate_all_experts:
                # Feed all tokens through this expert so the input quantizer sees
                # the full activation distribution; route-weight before adding so
                # the result still matches normal routing math.
                expert_output = self.experts[expert_idx](hidden_states)
                if has_tokens:
                    expert_weights = topk_weights[token_indices, weight_indices]
                    routed_output = expert_output[
                        token_indices
                    ] * expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(0, token_indices, routed_output)
            else:
                if has_tokens:
                    expert_input = hidden_states[token_indices]
                    expert_output = self.experts[expert_idx](expert_input)
                    expert_weights = topk_weights[token_indices, weight_indices]
                    routed_output = expert_output * expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(0, token_indices, routed_output)

        hidden_states = final_hidden_states.type(hidden_states.dtype).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class SequentialMistral4Experts(torch.nn.ModuleList):
    """
    Un-fuse ``Mistral4NaiveMoe``'s fused 3D expert tensors into per-expert
    ``Mistral4MLP`` modules so ``targets="Linear"`` can quantize each expert.

    Fused storage::

        gate_up_proj: (num_experts, 2 * intermediate_dim, hidden_dim)
        down_proj:    (num_experts, hidden_dim, intermediate_dim)

    The model computes ``F.linear(x, gate_up_proj[i])`` = ``x @ gate_up_proj[i].T``,
    so each per-expert slice is already in PyTorch ``nn.Linear`` ``(out, in)``
    layout — no transpose needed (unlike Llama 4). The ``2 * intermediate_dim``
    axis splits into gate (first half) and up (second half).
    """

    def __init__(self, config: Mistral4Config, original: Mistral4NaiveMoe):
        from transformers.models.mistral4.modeling_mistral4 import Mistral4MLP

        num_experts = original.num_experts
        intermediate_dim = original.intermediate_dim

        with skip_weights_initialize():
            super().__init__(
                [
                    Mistral4MLP(config, intermediate_size=intermediate_dim)
                    for _ in range(num_experts)
                ]
            )

        for i in range(num_experts):
            gate_up = original.gate_up_proj[i]  # (2 * intermediate_dim, hidden_dim)
            down = original.down_proj[i]  # (hidden_dim, intermediate_dim)

            gate_proj, up_proj = gate_up.chunk(2, dim=0)
            self[i].gate_proj.weight.data = gate_proj.contiguous()
            self[i].up_proj.weight.data = up_proj.contiguous()
            self[i].down_proj.weight.data = down.contiguous()

        # Free the now-dead fused tensors (matches the Llama 4 / Qwen3.5 pattern).
        original.gate_up_proj = None
        original.down_proj = None
