"""
Adapter utilities for MoNE pruning on non-linearized expert modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from llmcompressor.modifiers.pruning.mone.utils import MoNEStatsTracker
from llmcompressor.modifiers.pruning.reap import utils as reap_utils
from llmcompressor.modifiers.pruning.reap.utils import MoeModelAttrs, get_moe_attrs

__all__ = [
    "NativeFP8ExpertsMoNEAdapter",
    "get_mone_moe_attrs",
    "is_native_fp8_experts",
]


ROUTER_ATTRS = ("router", "gate")
EXPERTS_ATTRS = ("experts",)


def get_mone_moe_attrs(model: nn.Module, ignore: list[str]) -> MoeModelAttrs:
    """
    Find MoE layers supported by MoNE.

    The ordinary REAP/MoNE path handles linearized ``LinearExperts2D`` modules.
    Some native-FP8 checkpoints keep experts as Transformers ``FP8Experts``; for
    those, MoNE collects stats by temporarily wrapping the native forward.
    """

    fp8_attrs = _get_native_fp8_moe_attrs(model, ignore)
    if fp8_attrs is not None:
        return fp8_attrs

    return get_moe_attrs(model, ignore)


def is_native_fp8_experts(module: nn.Module) -> bool:
    fp8_experts_cls = _fp8_experts_cls()
    return fp8_experts_cls is not None and isinstance(module, fp8_experts_cls)


@dataclass
class NativeFP8ExpertsMoNEAdapter:
    """
    MoNE adapter for Transformers native ``FP8Experts`` modules.

    ``FP8Experts`` already computes each selected expert independently before
    applying router weights. MoNE needs exactly that unweighted expert output,
    so the calibration wrapper mirrors the upstream forward and inserts tracker
    updates at that point.
    """

    model: nn.Module
    layer_name: str
    experts_attr: str

    def __post_init__(self):
        self._original_forward: Callable | None = None

    @property
    def moe_block(self) -> nn.Module:
        return self.model.get_submodule(self.layer_name)

    @property
    def experts(self) -> nn.Module:
        return getattr(self.moe_block, self.experts_attr)

    def install_calibration(self, tracker: MoNEStatsTracker) -> None:
        experts = self.experts
        if self._original_forward is not None:
            return

        self._original_forward = experts.forward

        def calibration_forward(hidden_states, top_k_index, top_k_weights):
            return _native_fp8_forward(
                experts,
                hidden_states,
                top_k_index,
                top_k_weights,
                tracker=tracker,
            )

        experts.forward = calibration_forward

    def remove_calibration(self) -> None:
        if self._original_forward is None:
            return
        self.experts.forward = self._original_forward
        self._original_forward = None

    def apply_novices(
        self,
        novice_indices: list[int],
        mean_outputs: torch.Tensor,
        token_counts: torch.Tensor,
        zero_out_novice: bool,
        enable_novice_evolving: bool,
    ) -> list[int]:
        experts = self.experts
        hidden_size = mean_outputs.shape[-1]
        dtype = _model_compute_dtype(self.model)
        device = _first_module_device(experts, torch.device("cpu"))
        novice_set = {int(expert_idx) for expert_idx in novice_indices}
        constants = []
        init_tokens = []

        for expert_idx in novice_indices:
            value = (
                torch.zeros(hidden_size, dtype=torch.float32)
                if zero_out_novice
                else mean_outputs[expert_idx]
            )
            constants.append(value.to(device=device, dtype=dtype))
            init_tokens.append(
                int(token_counts[expert_idx].item()) if enable_novice_evolving else 0
            )

        constant_values = (
            torch.stack(constants)
            if constants
            else torch.empty(0, hidden_size, device=device, dtype=dtype)
        )
        constant_to_row = {
            int(expert_idx): row for row, expert_idx in enumerate(novice_indices)
        }

        experts.register_buffer(
            "constant_expert_values",
            constant_values,
            persistent=False,
        )
        experts.constant_expert_to_row = constant_to_row
        experts.constant_expert_ids = tuple(int(idx) for idx in novice_indices)
        experts.layer_idx = _config_layer_idx(self.layer_name)
        experts.approximate_expert_init_tokens = init_tokens

        def mone_forward(hidden_states, top_k_index, top_k_weights):
            return _native_fp8_forward(
                experts,
                hidden_states,
                top_k_index,
                top_k_weights,
                novice_set=novice_set,
                constant_values=experts.constant_expert_values,
                constant_to_row=experts.constant_expert_to_row,
                novice_init_tokens=experts.approximate_expert_init_tokens,
            )

        experts.forward = mone_forward
        return novice_indices


def _native_fp8_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    tracker: MoNEStatsTracker | None = None,
    novice_set: set[int] | None = None,
    constant_values: torch.Tensor | None = None,
    constant_to_row: dict[int, int] | None = None,
    novice_init_tokens: list[int] | None = None,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(
        hidden_states,
        dtype=torch.float32,
    )

    with torch.no_grad():
        if tracker is not None:
            tracker.update_routing(top_k_index, top_k_weights)

        expert_mask = F.one_hot(
            top_k_index,
            num_classes=experts.num_experts,
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = (
            torch.greater(
                expert_mask.sum(dim=(-1, -2)),
                0,
            )
            .nonzero(as_tuple=False)
            .view(-1)
        )

    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())

        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue

        if novice_set is not None and expert_idx in novice_set:
            if constant_values is None or constant_to_row is None:
                raise RuntimeError("Native FP8 MoNE constants were not initialized")
            row = constant_to_row[expert_idx]
            proj_out = constant_values[row].to(dtype=hidden_states.dtype)
            if novice_init_tokens is not None and novice_init_tokens[row] > 0:
                with torch.no_grad():
                    total = novice_init_tokens[row] + token_idx.numel()
                    constant_values[row].mul_(novice_init_tokens[row] / total)
                    constant_values[row].add_(
                        hidden_states[token_idx]
                        .float()
                        .sum(dim=0)
                        .to(device=constant_values.device, dtype=constant_values.dtype)
                        / total
                    )
                    novice_init_tokens[row] = total
                proj_out = constant_values[row].to(dtype=hidden_states.dtype)
            proj_out = proj_out.expand(token_idx.numel(), -1)
        else:
            current_state = hidden_states[token_idx]
            gate_up_act_scale = (
                experts.gate_up_proj_activation_scale[expert_idx]
                if experts.activation_scheme == "static"
                else None
            )
            proj_out = experts.linear(
                current_state,
                experts.gate_up_proj[expert_idx]
                if experts.has_gate
                else experts.up_proj[expert_idx],
                experts.gate_up_proj_scale_inv[expert_idx]
                if experts.has_gate
                else experts.up_proj_scale_inv[expert_idx],
                activation_scale=gate_up_act_scale,
            )
            proj_out = (
                experts._apply_gate(proj_out)
                if experts.has_gate
                else experts.act_fn(proj_out)
            )
            down_act_scale = (
                experts.down_proj_activation_scale[expert_idx]
                if experts.activation_scheme == "static"
                else None
            )
            proj_out = experts.linear(
                proj_out,
                experts.down_proj[expert_idx],
                experts.down_proj_scale_inv[expert_idx],
                activation_scale=down_act_scale,
            )

            if tracker is not None:
                tracker.update_expert(expert_idx, proj_out)

        routing_weights = top_k_weights[token_idx, top_k_pos, None]
        weighted_out = proj_out * routing_weights.to(proj_out.dtype)
        final_hidden_states.index_add_(
            0,
            token_idx,
            weighted_out.to(final_hidden_states.dtype),
        )

    return final_hidden_states.to(hidden_states.dtype)


def _get_native_fp8_moe_attrs(
    model: nn.Module,
    ignore: list[str],
) -> MoeModelAttrs | None:
    config = getattr(model, "config", None)
    if config is None:
        return None

    has_text_config = hasattr(config, "text_config")
    config = config.text_config if has_text_config else config
    num_experts_config_key, num_experts = _first_config_value(
        config,
        reap_utils.NUM_EXPERTS_CONFIG_KEYS,
    )
    if num_experts_config_key is None:
        return None

    _, top_k = _first_config_value(config, reap_utils.TOP_K_CONFIG_KEYS)
    if top_k is None:
        return None

    _, n_group = _first_config_value(config, reap_utils.N_GROUP_CONFIG_KEYS)
    _, top_k_group = _first_config_value(config, reap_utils.TOP_K_GROUP_CONFIG_KEYS)
    if (n_group is None) != (top_k_group is None):
        return None

    group_size = None
    if n_group is not None:
        if num_experts % n_group != 0:
            return None
        group_size = num_experts // n_group

    router_attr = None
    experts_attr = None
    for _, module in model.named_modules():
        for candidate_experts_attr in EXPERTS_ATTRS:
            experts = getattr(module, candidate_experts_attr, None)
            if not is_native_fp8_experts(experts):
                continue
            for candidate_router_attr in ROUTER_ATTRS:
                if hasattr(module, candidate_router_attr):
                    router_attr = candidate_router_attr
                    experts_attr = candidate_experts_attr
                    break
            if router_attr is not None:
                break
        if router_attr is not None:
            break

    if router_attr is None or experts_attr is None:
        return None

    moe_layer_names = []
    for name, module in model.named_modules():
        if any(re.search(pattern, name) for pattern in ignore):
            continue
        if hasattr(module, router_attr) and is_native_fp8_experts(
            getattr(module, experts_attr, None)
        ):
            moe_layer_names.append(name)

    if not moe_layer_names:
        return None

    logger.info(
        f"Found {len(moe_layer_names)} MoE layers with native FP8Experts format"
    )
    return MoeModelAttrs(
        num_experts_config_key=num_experts_config_key,
        router_attr=router_attr,
        experts_attr=experts_attr,
        moe_layer_names=moe_layer_names,
        num_experts=int(num_experts),
        top_k=int(top_k),
        has_text_config=has_text_config,
        n_group=n_group,
        top_k_group=top_k_group,
        group_size=group_size,
    )


def _fp8_experts_cls() -> type[nn.Module] | None:
    try:
        from transformers.integrations.finegrained_fp8 import FP8Experts
    except Exception:
        return None
    return FP8Experts


def _first_config_value(config, keys: list[str]) -> tuple[str | None, int | None]:
    for key in keys:
        if hasattr(config, key):
            return key, getattr(config, key)
    return None, None


def _config_layer_idx(layer_name: str) -> int:
    parts = layer_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers" and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    raise ValueError(f"Could not parse layer index from MoE layer name {layer_name}")


def _model_compute_dtype(model: nn.Module) -> torch.dtype:
    config = getattr(model, "config", None)
    for attr in ("torch_dtype", "dtype"):
        value = getattr(config, attr, None)
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str) and hasattr(torch, value):
            dtype = getattr(torch, value)
            if isinstance(dtype, torch.dtype) and not str(dtype).startswith(
                "torch.float8"
            ):
                return dtype

    for param in model.parameters():
        if param.is_floating_point() and not str(param.dtype).startswith(
            "torch.float8"
        ):
            return param.dtype
    return torch.bfloat16


def _first_module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return fallback
