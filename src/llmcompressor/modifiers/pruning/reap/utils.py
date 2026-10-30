"""
Utilities for REAP: expert pruning.

MoE layer detection is shared with the other expert pruning modifiers and
lives in ``llmcompressor.modifiers.pruning.utils``; it is re-exported here
for backwards compatibility.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from compressed_tensors import align_module_device
from compressed_tensors.offload.utils import to_tensor

from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers.pruning.utils import (
    NUM_EXPERTS_CONFIG_KEYS,
    NUM_EXPERTS_MODULE_KEYS,
    MoeModelAttrs,
    get_moe_attrs,
)

__all__ = [
    "REAP_PRECISION",
    "MoeModelAttrs",
    "NUM_EXPERTS_CONFIG_KEYS",
    "get_moe_attrs",
    "prune_moe_layer",
]


REAP_PRECISION = torch.float32


def prune_moe_layer(
    model: nn.Module,
    layer_name: str,
    retained: list[int],
    moe_attrs: MoeModelAttrs,
) -> list[int]:
    """
    Structurally prune a MoE block to keep only ``retained`` experts: slice the
    expert ``ModuleList``, shrink the router, and update expert-count attributes.
    Offload-safe: experts are kept as existing module objects (offload state
    travels with them) and the small router is resized under
    ``align_module_device``.
    """
    moe_block = model.get_submodule(layer_name)
    router = getattr(moe_block, moe_attrs.router_attr)
    experts = getattr(moe_block, moe_attrs.experts_attr)

    # Preserve non-expert modules (e.g., act_fn in LinearExperts2D)
    # These are modules that are not instances of ExpertMLP subclasses
    non_expert_modules = {}
    for key, module in experts._modules.items():
        if not isinstance(module, ExpertMLP):
            non_expert_modules[key] = module

    # Rebuild with retained experts
    new_modules = OrderedDict(
        ((str(i), experts[pos]) for i, pos in enumerate(retained))
    )

    # Re-add non-expert modules
    new_modules.update(non_expert_modules)

    experts._modules = new_modules
    experts.num_experts = len(retained)

    _prune_router(router, retained)

    # Some models (e.g. Hy3) place e_score_correction_bias on the MoE block
    # rather than on the router; prune it here
    e_bias = getattr(moe_block, "e_score_correction_bias", None)
    if e_bias is not None:
        e_bias = to_tensor(e_bias[retained].contiguous(), e_bias)
        moe_block.e_score_correction_bias = e_bias

    # Update num_experts for any other modules in the layer that may track it
    for holder in (moe_block, router):
        for key in NUM_EXPERTS_MODULE_KEYS:
            if isinstance(getattr(holder, key, None), int):
                setattr(holder, key, len(retained))

    return retained


def _prune_router(router: nn.Module, retained: list[int]):
    router_weight: torch.nn.Parameter | None = getattr(router, "weight", None)
    router_bias: torch.nn.Parameter | None = getattr(router, "bias", None)
    router_escore: torch.nn.Parameter | None = getattr(
        router, "e_score_correction_bias", None
    )
    router_out_features: int | None = getattr(router, "out_features", None)
    if not isinstance(router_weight, nn.Parameter):
        raise ValueError("Unexpected MoE router. Router must have `weight` attribute")

    retained_t = torch.tensor(retained, dtype=torch.long)

    with align_module_device(router):
        retained_t = retained_t.to(router_weight.device)
        new_weight = router_weight.detach()[retained_t].contiguous()
        new_bias = (
            router_bias.detach()[retained_t].contiguous()
            if router_bias is not None
            else None
        )
        new_escore = (
            router_escore.detach()[retained_t].contiguous()
            if router_escore is not None
            else None
        )

    # Direct attribute assignment replaces a parameter/buffer with a different
    # shape and is correct for both offloaded modules (routed through the
    # OffloadCache, which re-offloads the new shape) and ordinary modules.
    router.weight = torch.nn.Parameter(
        new_weight, requires_grad=router_weight.requires_grad
    )
    if new_bias is not None:
        router.bias = torch.nn.Parameter(new_bias, requires_grad=False)
    if new_escore is not None:
        router.e_score_correction_bias = torch.nn.Parameter(
            new_escore, requires_grad=False
        )

    if router_out_features is not None:
        router.out_features = len(retained)
