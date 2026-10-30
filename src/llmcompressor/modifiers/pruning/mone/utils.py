"""
Utilities for MoNe (Mixture of Novices): expert collapse.

The lowest-saliency experts are *collapsed* into constant "novice" vectors
(each expert's mean output, an unbiased estimator of its contribution).
Retained experts are left-shifted while the router keeps all ``R + N`` rows,
so novices continue to receive their original router weights at inference
time.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from compressed_tensors.offload import get_execution_device

from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers.pruning.utils import (
    NUM_EXPERTS_CONFIG_KEYS,
    MoeModelAttrs,
    _prune_router,
)

__all__ = ["collapse_moe_layer"]


def collapse_moe_layer(
    model: nn.Module,
    layer_name: str,
    retained: list[int],
    novices: torch.Tensor,
    moe_attrs: MoeModelAttrs,
) -> list[int]:
    """
    Collapse a MoE block into retained experts + novices (MoNe).

    The retained experts are left-shifted into a contiguous ``ModuleList`` exactly
    as in REAP pruning. Unlike REAP, the router is *not* shrunk: its rows are
    reordered so the retained experts occupy rows ``0..R`` and the collapsed
    experts occupy rows ``R..R+N`` (matching the order of ``novices``). The
    collapsed experts' mean outputs are installed as ``experts.novices``.

    :param novices: ``(N, hidden_dim)`` tensor of novice vectors, ordered to match
        the dropped experts (i.e. the router rows that follow the retained block)
    """
    moe_block = model.get_submodule(layer_name)
    router = getattr(moe_block, moe_attrs.router_attr)
    experts = getattr(moe_block, moe_attrs.experts_attr)

    num_experts = moe_attrs.num_experts
    retained_set = set(retained)
    dropped = [i for i in range(num_experts) if i not in retained_set]
    assert novices.shape[0] == len(dropped), (
        f"collapse_moe_layer: got {novices.shape[0]} novices but "
        f"{len(dropped)} experts were collapsed"
    )

    # Preserve non-expert modules (e.g. act_fn in LinearExperts2D)
    non_expert_modules = {
        key: module
        for key, module in experts._modules.items()
        if not isinstance(module, ExpertMLP)
    }

    # Rebuild with retained experts (left-shifted), re-adding non-expert modules
    new_modules = OrderedDict((str(i), experts[pos]) for i, pos in enumerate(retained))
    new_modules.update(non_expert_modules)
    experts._modules = new_modules
    experts.num_experts = len(retained)

    # Narrow the repack source config so to_experts_module() rebuilds only the
    # retained experts; the router and novices keep the full R + N width.
    _narrow_source_config(experts, len(retained))

    # Install novice vectors on the experts module.
    _set_novices(experts, novices)

    # Reorder router rows to [retained..., dropped...] while keeping all R + N rows.
    _prune_router(router, retained + dropped)

    # Reorder a MoE-block-level score-correction bias if present (e.g. Hy3).
    e_bias = getattr(moe_block, "e_score_correction_bias", None)
    if e_bias is not None:
        from compressed_tensors.offload.utils import to_tensor

        order = torch.tensor(retained + dropped, dtype=torch.long, device=e_bias.device)
        moe_block.e_score_correction_bias = to_tensor(
            e_bias[order].contiguous(), e_bias
        )

    return retained


def _set_novices(experts: nn.Module, novices: torch.Tensor) -> None:
    exec_device = get_execution_device(experts)
    target_dtype = novices.dtype
    for param in experts.parameters():
        if param.is_floating_point():
            target_dtype = param.dtype
            break
    novices = novices.to(device=exec_device, dtype=target_dtype).contiguous()
    experts.register_parameter("novices", nn.Parameter(novices, requires_grad=False))


def _narrow_source_config(experts: nn.Module, new_num_experts: int) -> None:
    config = getattr(experts, "_source_config", None)
    if config is None:
        return
    new_config = copy.copy(config)
    for key in NUM_EXPERTS_CONFIG_KEYS:
        if hasattr(new_config, key):
            setattr(new_config, key, new_num_experts)
    experts._source_config = new_config
