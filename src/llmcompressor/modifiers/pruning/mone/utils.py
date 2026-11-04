"""
Utilities for MoNe (Mixture of Novices): expert collapse.

The lowest-saliency experts are *collapsed* into constant "novice" vectors
(each expert's mean output, an unbiased estimator of its contribution).
Retained experts are left-shifted while the router keeps all ``R + N`` rows,
so novices continue to receive their original router weights at inference
time.
"""

import torch
import torch.nn as nn
from compressed_tensors.offload import get_execution_device
from compressed_tensors.offload.utils import to_tensor

from llmcompressor.modifiers.pruning.expert_sparsity.utils import (
    get_experts_router,
    prune_moe_layer,
    reorder_router,
)

__all__ = ["mone_collapse_moe_layer", "set_novices"]


def mone_collapse_moe_layer(
    moe_name: str,
    moe_module: nn.Module,
    retained: list[int],
    dropped: list[int],
    novices: torch.Tensor,
) -> list[int]:
    """
    Collapse a MoE block into retained experts + novices (MoNe).

    The retained experts are left-shifted into a contiguous ``ModuleList``
    exactly as in plain expert pruning (via
    :func:`~llmcompressor.modifiers.pruning.expert_sparsity.utils.prune_moe_layer`
    with ``prune_router=False``). Unlike plain pruning, the router is *not*
    shrunk: its rows are reordered so the retained experts occupy rows
    ``0..R-1`` and the collapsed experts occupy rows ``R..R+N-1`` (matching
    the order of ``novices``). The collapsed experts' mean outputs are
    installed as ``experts.novices``, which the experts module serves in place
    of the collapsed experts' outputs for tokens the router sends there.

    :param moe_name: name of the MoE block, used for error reporting
    :param moe_module: the MoE block module
    :param retained: indices of the experts to keep, in their new order
    :param dropped: indices of the experts to collapse, ordered to match the
        rows of ``novices``
    :param novices: ``(N, hidden_dim)`` tensor of novice vectors, ordered to
        match ``dropped`` (i.e. the router rows that follow the retained
        block)
    :return: the retained expert indices
    """
    experts, router = get_experts_router(moe_name, moe_module)

    assert novices.shape[0] == len(dropped), (
        f"mone_collapse_moe_layer: got {novices.shape[0]} novices but "
        f"{len(dropped)} experts were collapsed"
    )
    assert len(retained) + len(dropped) == router.weight.shape[0], (
        f"mone_collapse_moe_layer: {len(retained)} retained + {len(dropped)} dropped "
        f"experts do not cover the router's {router.weight.shape[0]} rows"
    )

    # Left-shift the retained experts without touching the router: it is
    # reordered below so the novices keep receiving their routing weights
    prune_moe_layer(moe_name, moe_module, retained, prune_router=False)

    # Install novice vectors on the experts module
    set_novices(experts, novices)

    # Reorder router rows to [retained..., dropped...] while keeping all
    # R + N rows
    reorder_router(router, retained + dropped)

    # Reorder a MoE-block-level score-correction bias if present (e.g. Hy3),
    # in lockstep with the router rows
    e_bias = getattr(moe_module, "e_score_correction_bias", None)
    if e_bias is not None:
        order = torch.tensor(retained + dropped, dtype=torch.long, device=e_bias.device)
        moe_module.e_score_correction_bias = to_tensor(
            e_bias[order].contiguous(), e_bias
        )

    return retained


def set_novices(experts: nn.Module, novices: torch.Tensor) -> None:
    exec_device = get_execution_device(experts)
    target_dtype = novices.dtype
    for param in experts.parameters():
        if param.is_floating_point():
            target_dtype = param.dtype
            break
    novices = novices.to(device=exec_device, dtype=target_dtype).contiguous()
    experts.register_parameter("novices", nn.Parameter(novices, requires_grad=False))
