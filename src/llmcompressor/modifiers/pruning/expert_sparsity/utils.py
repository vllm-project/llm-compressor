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
    "prune_router",
]


ROUTER_ATTRS = ["router", "gate"]


def get_experts_router(
    moe_name: str, moe_module: torch.nn.Module
) -> tuple[LinearExperts2D | None, torch.nn.Module | None]:
    (
        experts,
        router,
    ) = None, None
    for name, child in moe_module.named_children():
        if isinstance(child, LinearExperts2D):
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
) -> list[int]:
    """
    Structurally prune a MoE block to keep only ``retained`` experts: slice the
    expert ``ModuleList``, shrink the router, and update expert-count attributes.
    Offload-safe: experts are kept as existing module objects (offload state
    travels with them) and the small router is resized under
    ``align_module_device``.
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

    # Rebuild router with retained experts
    prune_router(router, retained)

    # Some models (e.g., Hy3) place e_score_correction_bias on the MoE block
    # rather than on the router; prune it here
    e_bias = getattr(moe_module, "e_score_correction_bias", None)
    if e_bias is not None:
        e_bias = to_tensor(e_bias[retained].contiguous(), e_bias)
        moe_module.e_score_correction_bias = e_bias

    # Update num_experts for any other modules in the layer that may track it
    for holder in (moe_module, experts, router):
        for key in ["num_experts", "n_experts", "n_routed_experts"]:
            if isinstance(getattr(holder, key, None), int):
                setattr(holder, key, len(retained))

    return retained


def prune_router(router: nn.Module, retained: list[int]):
    router_weight: torch.nn.Parameter | None = getattr(router, "weight", None)
    router_bias: torch.nn.Parameter | None = getattr(router, "bias", None)
    router_escore: torch.nn.Parameter | None = getattr(
        router, "e_score_correction_bias", None
    )
    router_out_features: int | None = getattr(router, "out_features", None)
    if not isinstance(router_weight, nn.Parameter):
        raise ValueError("Unexpected MoE router. Router must have `weight` attribute")

    retained_t = torch.tensor(retained, dtype=torch.long, device=router_weight.device)

    new_weight = router_weight.detach()[retained_t].contiguous()
    router.weight = torch.nn.Parameter(
        new_weight, requires_grad=router_weight.requires_grad
    )
    if router_bias is not None:
        new_bias = router_bias.detach()[retained_t].contiguous()
        router.bias = torch.nn.Parameter(
            new_bias, requires_grad=router_bias.requires_grad
        )

    if router_escore is not None:
        new_escore = router_escore.detach()[retained_t].contiguous()
        router.e_score_correction_bias = torch.nn.Parameter(
            new_escore, requires_grad=router_escore.requires_grad
        )
    if router_out_features is not None:
        router.out_features = len(retained)
