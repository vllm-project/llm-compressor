from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from llmcompressor.modeling.moe_context import MoECalibrationModule


class LinearExpert(nn.Module):
    """
    One MoE expert with separate gate / up / down projections.

    This mirrors the GPT-OSS expert behavior:
        gate = clamp(gate_proj(x))
        up   = clamp(up_proj(x))
        glu  = gate * sigmoid(alpha * gate)
        y    = down_proj((up + 1) * glu)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        alpha: float,
        limit: float,
    ):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)

        glu = gate * torch.sigmoid(self.alpha * gate)
        act = (up + 1) * glu
        return self.down_proj(act)


class LinearExperts(nn.Module):
    """
    Container of multiple LinearExpert modules, driven by
    router_indices / routing_weights.

    This is the "separate gate/up" layout.
    It is meant to replace the original GPT-OSS `experts` submodule.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        alpha: float = 1.702,
        limit: float = 7.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_dim = intermediate_size
        self.num_experts = num_experts
        self.alpha = alpha
        self.limit = limit

        self.experts = nn.ModuleList(
            [
                LinearExpert(hidden_size, intermediate_size, alpha, limit)
                for _ in range(num_experts)
            ]
        )

    @torch.no_grad()
    def copy_from_fused_weights(
        self,
        legacy_gate_up_W: torch.Tensor,  # [E, H, 2D]
        legacy_gate_up_b: torch.Tensor,  # [E, 2D]
        legacy_down_W: torch.Tensor,  # [E, D, H]
        legacy_down_b: torch.Tensor,  # [E, H]
    ) -> None:
        """
        De-interleave fused gate_up weights/bias and copy into separate gate/up experts.
        """
        E, H, twoD = legacy_gate_up_W.shape
        assert E == self.num_experts
        D = twoD // 2
        assert D == self.expert_dim

        for i in range(E):
            Wi = legacy_gate_up_W[i]  # [H, 2D]
            bi = legacy_gate_up_b[i]  # [2D]

            Wg = Wi[:, 0::2].contiguous()  # [H, D]
            Wu = Wi[:, 1::2].contiguous()  # [H, D]
            bg = bi[0::2].contiguous()  # [D]
            bu = bi[1::2].contiguous()  # [D]

            expert = self.experts[i]
            expert.gate_proj.weight.copy_(Wg.t())
            expert.gate_proj.bias.copy_(bg)
            expert.up_proj.weight.copy_(Wu.t())
            expert.up_proj.bias.copy_(bu)

            expert.down_proj.weight.copy_(legacy_down_W[i].t())
            expert.down_proj.bias.copy_(legacy_down_b[i])

    def _normalize_shapes(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ):
        """Normalize input shapes to 2D format for processing."""
        # Normalize shapes to [tokens, H], [tokens, top_k], [tokens, E]
        if hidden_states.dim() == 3:
            B, T, H = hidden_states.shape
            x = hidden_states.reshape(-1, H)
        else:
            # Already flattened
            B, _ = 1, hidden_states.shape[0]
            H = hidden_states.shape[-1]
            x = hidden_states

        if router_indices.dim() == 3:
            router_indices = router_indices.reshape(
                -1, router_indices.shape[-1]
            )
        if routing_weights.dim() == 3:
            routing_weights = routing_weights.reshape(
                -1, routing_weights.shape[-1]
            )

        return x, router_indices, routing_weights, B, H

    def _route_and_compute(
        self,
        x: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        calibrate_all: bool = False,
    ) -> torch.Tensor:
        """Shared routing logic for expert computation."""
        num_experts_plus_dummy = routing_weights.shape[1]
        out = torch.zeros_like(x)

        # GPT-OSS router uses an extra "no expert" bucket at index E
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_indices, num_classes=num_experts_plus_dummy
            ).permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for idx in expert_hit:
            e = idx[0].item()
            if e == self.num_experts:
                # Skip "no expert" bucket
                continue

            _, token_idx = torch.where(expert_mask[e])
            expert = self.experts[e]

            if calibrate_all:
                # Process all tokens through expert for calibration
                yi = expert(x)[token_idx]
            else:
                # Normal routing: only process assigned tokens
                xi = x[token_idx]
                yi = expert(xi)

            w = routing_weights[token_idx, e, None]
            out.index_add_(0, token_idx, (yi * w).to(out.dtype))

        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implements the MoE computation using the router outputs.

        This is compatible with the GPT-OSS MoE call pattern:
            experts(hidden_states, router_indices, routing_weights)
        """
        assert (
            routing_weights is not None and router_indices is not None
        ), "router inputs required"

        x, router_indices, routing_weights, B, H = self._normalize_shapes(
            hidden_states, router_indices, routing_weights
        )

        out = self._route_and_compute(x, router_indices, routing_weights)
        return out.view(B, -1, H)


@MoECalibrationModule.register("GptOssExperts")
class CalibrationLinearExperts(LinearExperts, MoECalibrationModule):
    """
    Calibration version of LinearExperts that sends all tokens to all experts.

    This module wraps the already-linearized LinearExperts to provide
    calibration support during quantization. Since LinearExperts already has
    the correct structure (separate gate/up/down projections), just add the
    calibrate_all_experts functionality.
    """

    is_permanent = True

    def __init__(
        self,
        original: LinearExperts,
        config,
        calibrate_all_experts: bool = True,
    ):
        # Don't call LinearExperts.__init__, just copy attributes
        nn.Module.__init__(self)
        self.hidden_size = original.hidden_size
        self.expert_dim = original.expert_dim
        self.num_experts = original.num_experts
        self.alpha = original.alpha
        self.limit = original.limit
        self.experts = original.experts
        self.calibrate_all_experts = calibrate_all_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implements the MoE computation using the router outputs.

        This is compatible with the GPT-OSS MoE call pattern:
            experts(hidden_states, router_indices, routing_weights)

        When calibrate_all_experts=True, all experts process all tokens
        to ensure proper calibration statistics. Enables activations
        through all expert paths.
        """
        assert (
            routing_weights is not None and router_indices is not None
        ), "router inputs required"

        x, router_indices, routing_weights, B, H = self._normalize_shapes(
            hidden_states, router_indices, routing_weights
        )

        out = self._route_and_compute(
            x, router_indices, routing_weights, self.calibrate_all_experts
        )
        return out.view(B, -1, H)


@dataclass
class ExpertMeta:
    path: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    device: torch.device
    dtype: torch.dtype


def get_module_by_path(root: nn.Module, dotpath: str) -> nn.Module:
    m: nn.Module = root
    if not dotpath:
        return root
    for p in dotpath.split("."):
        m = getattr(m, p)
    return m


def set_module_by_path(
    root: nn.Module, dotpath: str, new_module: nn.Module
) -> None:
    parts = dotpath.split(".")
    parent = get_module_by_path(root, ".".join(parts[:-1]))
    setattr(parent, parts[-1], new_module)


def find_experts(model: nn.Module) -> List[ExpertMeta]:
    """
    Locate GPT-OSS MoE expert modules under model.model.layers[*].mlp.experts.
    """
    metas: List[ExpertMeta] = []
    for li, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        device = next(experts.parameters(), torch.zeros(())).device
        dtype = next(experts.parameters(), torch.zeros(())).dtype
        intermediate = getattr(experts, "expert_dim", None)
        if intermediate is None:
            intermediate = getattr(experts, "intermediate_size")

        metas.append(
            ExpertMeta(
                path=f"model.layers.{li}.mlp.experts",
                hidden_size=experts.hidden_size,
                intermediate_size=intermediate,
                num_experts=experts.num_experts,
                device=device,
                dtype=dtype,
            )
        )
    return metas


def convert_model_for_quantization_gptoss(
    model: nn.Module, calibrate_all_experts: bool = True
) -> None:
    """
    In-place conversion of a GPT-OSS model for quantization.

    This function performs two key transformations:
    1. Linearizes fused MoE expert blocks (gate_up_proj/down_proj) into
       separate nn.Linear parameters (gate_proj, up_proj, down_proj)
    2. Wraps them with CalibrationLinearExperts for proper calibration

    Args:
        model: The GPT-OSS model to convert (modified in-place)
        calibrate_all_experts: If True, all experts will see all tokens
            during calibration. This is the recommended setting for proper
            quantization statistics. Set to False only if you want normal
            routing behavior during calibration.
    """
    metas = find_experts(model)
    for meta in metas:
        legacy = get_module_by_path(model, meta.path)

        # Sanity check that this is the fused layout we expect.
        if not all(
            hasattr(legacy, attr)
            for attr in [
                "gate_up_proj",
                "gate_up_proj_bias",
                "down_proj",
                "down_proj_bias",
            ]
        ):
            continue

        # Step 1: Create LinearExperts with separate gate/up/down projections
        linear_experts = LinearExperts(
            hidden_size=meta.hidden_size,
            intermediate_size=meta.intermediate_size,
            num_experts=meta.num_experts,
        ).to(device=meta.device, dtype=meta.dtype)

        linear_experts.copy_from_fused_weights(
            legacy_gate_up_W=legacy.gate_up_proj,
            legacy_gate_up_b=legacy.gate_up_proj_bias,
            legacy_down_W=legacy.down_proj,
            legacy_down_b=legacy.down_proj_bias,
        )

        # Step 2: Wrap with CalibrationLinearExperts for MoE calibration
        calibration_experts = CalibrationLinearExperts(
            original=linear_experts,
            config=model.config,
            calibrate_all_experts=calibrate_all_experts,
        )

        set_module_by_path(model, meta.path, calibration_experts)
