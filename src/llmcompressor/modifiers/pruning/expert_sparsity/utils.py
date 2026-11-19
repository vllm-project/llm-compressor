"""
Utilities shared by the expert pruning modifiers: MoE layer detection and
structural expert pruning.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from compressed_tensors.offload.utils import to_tensor

from llmcompressor.modeling.moe.linear_experts import ExpertMLP, LinearExperts2D

__all__ = [
    "get_experts_router",
    "prune_moe_layer",
    "reorder_router",
]


ROUTER_ATTRS = ["router", "gate"]


def get_experts_router(
    moe_name: str, moe_module: torch.nn.Module
) -> tuple[LinearExperts2D | None, torch.nn.Module | None]:
    """
    Get the experts and router layers from an MoE layer

    :param moe_name: fqn of moe module
    :param moe_module: moe module with router and expert children
    """

    from llmcompressor.modeling.moe.gpt_oss import GptOssLinearExperts
    from llmcompressor.modeling.moe.llama4 import Llama4LinearExperts

    experts = None
    router = None
    for name, child in moe_module.named_children():
        if isinstance(child, (GptOssLinearExperts, Llama4LinearExperts)):
            raise ValueError(f"Found unsupported MoE layer {child.__class__}")
        elif isinstance(child, LinearExperts2D):
            experts = child
        elif name in ROUTER_ATTRS:
            router = child

    if experts is not None and router is not None:
        return experts, router

    elif experts is not None and router is None:
        raise ValueError(
            f"Could not find router. Please ensure that one of {moe_name} + "
            f"{ROUTER_ATTRS} exists or the router name is listed in "
            "`src/llmcompressor/modifiers/pruning/expert_sparsity/utils.py::ROUTER_ATTRS`"
        )

    return None, None


def prune_moe_layer(
    moe_name: str,
    moe_module: torch.nn.Module,
    retained: list[int],
    prune_router: bool = True,
) -> list[int]:
    """
    Structurally prune a MoE block to keep only ``retained`` experts: slice the
    expert ``ModuleList`` and update expert-count attributes. Offload-safe:
    experts are kept as existing module objects, so offload state travels with
    them.

    :param moe_name: name of the MoE block, used for error reporting
    :param moe_module: the MoE block module
    :param retained: indices of the experts to keep, in their new order
    :param prune_router: also shrink the router, any score-correction bias, and
        the router's expert-count attributes to the retained experts. When
        False, the router is left untouched for callers which keep routing to
        the dropped experts -- e.g. MoNe, which collapses them into novices and
        reorders the router rows itself
    :return: the retained expert indices
    """
    assert retained is not None, "Attempted to prune an moe that was ignored"
    experts, router = get_experts_router(moe_name, moe_module)

    # Preserve non-expert modules (e.g., act_fn in LinearExperts2D)
    # These are modules that are not instances of ExpertMLP subclasses
    non_expert_modules = {}
    for key, module in experts._modules.items():
        if not isinstance(module, ExpertMLP):
            non_expert_modules[key] = module

    # Rebuild experts with retained experts and non-expert mdoules
    new_modules = OrderedDict(
        ((str(i), experts[pos]) for i, pos in enumerate(retained))
    )
    new_modules.update(non_expert_modules)
    experts._modules = new_modules

    if prune_router:
        # Rebuild router with retained experts
        reorder_router(router, retained)

        # Some models (e.g., Hy3) place e_score_correction_bias on the MoE block
        # rather than on the router; prune it here
        e_bias = getattr(moe_module, "e_score_correction_bias", None)
        if e_bias is not None:
            e_bias = to_tensor(e_bias[retained].contiguous(), e_bias)
            moe_module.e_score_correction_bias = e_bias

    # Update num_experts for any other modules in the layer that may track it.
    # The router is excluded when it is not pruned: it keeps its full width so
    # the dropped experts stay routable
    holders = (moe_module, experts, router) if prune_router else (moe_module, experts)
    for holder in holders:
        for key in ["num_experts", "n_experts", "n_routed_experts"]:
            if isinstance(getattr(holder, key, None), int):
                setattr(holder, key, len(retained))

    return retained


# per-expert vectors (indexed by expert along dim 0) which some routers carry
# alongside their projection and which must be reordered with the router rows:
# score-correction biases (DeepSeek-V3, nemotron_h) and per-expert output
# scales (gemma4)
ROUTER_EXPERT_VECTOR_ATTRS = ["e_score_correction_bias", "per_expert_scale"]


def _get_router_linear(router: nn.Module) -> nn.Module:
    """
    The router's expert-dimension linear projection, whose rows are indexed by
    expert.

    Most routers expose it on the router module itself (``router.weight``,
    e.g. qwen2_moe, nemotron_h); others wrap a dedicated ``nn.Linear``
    submodule (e.g. gemma4's ``proj``). Returns the module carrying the
    ``weight`` (and optional ``bias``/``out_features``) to reorder.
    """
    if isinstance(getattr(router, "weight", None), nn.Parameter):
        return router
    for child in router.children():
        if isinstance(child, nn.Linear):
            return child
    raise ValueError(
        "Unexpected MoE router: expected a `weight` attribute or a child "
        f"`nn.Linear`, found neither on {type(router).__name__}"
    )


def _reorder_expert_vector(
    module: nn.Module, attr: str, retained_t: torch.Tensor
) -> None:
    """Reorder a module's per-expert vector ``attr`` (a Parameter or buffer
    indexed by expert along dim 0) by ``retained_t``, in place, preserving
    whether it was a Parameter or a buffer."""
    vector = getattr(module, attr, None)
    if not isinstance(vector, torch.Tensor):
        return
    reordered = vector.detach()[retained_t].contiguous()
    if isinstance(vector, nn.Parameter):
        new_value = nn.Parameter(reordered, requires_grad=vector.requires_grad)
    else:
        # registered buffer (e.g. nemotron_h's e_score_correction_bias)
        new_value = reordered
    setattr(module, attr, new_value)


def reorder_router(router: nn.Module, order: list[int]):
    """
    Reorder a router's rows to ``order``: its projection weight (plus any bias,
    score-correction bias, or per-expert scale) is indexed by ``order``, and
    ``out_features`` is set to ``len(order)``.

    The expert-dimension projection is resolved from the router module, which
    either exposes it directly (``router.weight``, e.g. qwen2_moe, nemotron_h)
    or wraps a dedicated ``nn.Linear`` (e.g. gemma4's ``proj``). Any per-expert
    vectors the router carries (``e_score_correction_bias``, gemma4's
    ``per_expert_scale``) are reordered alongside it.

    Passing a subset of the experts shrinks the router to those rows (plain
    expert pruning); passing a full permutation keeps its width while
    reordering the rows, so e.g. MoNe's novices can keep occupying the rows
    which follow the retained experts.
    """
    linear = _get_router_linear(router)
    router_weight: torch.nn.Parameter = linear.weight

    retained_t = torch.tensor(order, dtype=torch.long, device=router_weight.device)

    new_weight = router_weight.detach()[retained_t].contiguous()
    linear.weight = torch.nn.Parameter(
        new_weight, requires_grad=router_weight.requires_grad
    )
    _reorder_expert_vector(linear, "bias", retained_t)

    # per-expert vectors live on the router module itself
    for attr in ROUTER_EXPERT_VECTOR_ATTRS:
        _reorder_expert_vector(router, attr, retained_t)

    # update any declared output width, on both the projection and the router
    for module in {linear, router}:
        if getattr(module, "out_features", None) is not None:
            module.out_features = len(order)
