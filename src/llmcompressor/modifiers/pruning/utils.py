"""
Utilities shared by the expert pruning modifiers: MoE layer detection and
structural expert pruning.
"""

import re
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
from compressed_tensors import align_module_device
from compressed_tensors.offload.utils import to_tensor
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.modeling.moe.gpt_oss import GptOssLinearExperts
from llmcompressor.modeling.moe.linear_experts import ExpertMLP, LinearExperts2D
from llmcompressor.modeling.moe.llama4 import Llama4LinearExperts
from llmcompressor.utils.helpers import getattr_fallbacks

__all__ = [
    "MoeModelAttrs",
    "NUM_EXPERTS_CONFIG_KEYS",
    "NUM_EXPERTS_MODULE_KEYS",
    "build_moe_num_experts",
    "get_moe_attrs",
    "prune_moe_layer",
]


ROUTER_ATTRS = ["router", "gate"]
EXPERTS_ATTRS = ["experts"]
NUM_EXPERTS_CONFIG_KEYS = [
    "num_experts",
    "num_local_experts",
    "moe_num_experts",
    "n_routed_experts",
]
TOP_K_CONFIG_KEYS = ["num_experts_per_tok", "top_k", "moe_top_k"]
N_GROUP_CONFIG_KEYS = ["n_group"]
TOP_K_GROUP_CONFIG_KEYS = ["topk_group", "top_k_group"]
NUM_EXPERTS_MODULE_KEYS = ["num_experts", "n_experts", "n_routed_experts"]


@dataclass
class MoeModelAttrs:
    # moe config
    num_experts: int
    top_k: int

    # group-limited moe config
    n_group: int | None
    top_k_group: int | None
    group_size: int | None

    # attributes used to find MoE layers
    experts_attr: str
    router_attr: str

    # names of the model's MoE layers
    moe_layer_names: list[str]


def get_moe_attrs(model: PreTrainedModel, ignore: list[str]) -> MoeModelAttrs:
    config = model.config.get_text_config()

    num_experts = getattr_fallbacks(config, NUM_EXPERTS_CONFIG_KEYS)
    top_k = getattr_fallbacks(config, TOP_K_CONFIG_KEYS)
    n_group = getattr_fallbacks(config, N_GROUP_CONFIG_KEYS, default=None)
    top_k_group = getattr_fallbacks(config, TOP_K_GROUP_CONFIG_KEYS, default=None)

    # validate group-limited router configs
    if (n_group is None) != (top_k_group is None):
        attr_name = "n_group" if n_group is not None else "top_k_group"
        raise ValueError(
            f"Detected one group-limited router attribute ({attr_name} "
            "is set) but could not find a config attribute for the other. "
            "Make sure the name of the model config's n_group and "
            "top_k_group attributes are in N_GROUP_CONFIG_KEYS and "
            "TOP_K_GROUP_CONFIG_KEYS in pruning/utils.py"
        )
    group_size = None
    if n_group is not None:
        if num_experts % n_group != 0:
            raise ValueError(
                f"Group limited router detected, but {num_experts} experts "
                f"not divisible by n_group={n_group}"
            )
        group_size = num_experts // n_group

    # find the expert and router attribute names using the first match
    router_attr = None
    experts_attr = None
    for _, module in model.named_modules():
        for e_attr in EXPERTS_ATTRS:
            if hasattr(module, e_attr):
                for r_attr in ROUTER_ATTRS:
                    if hasattr(module, r_attr):
                        router_attr = r_attr
                        experts_attr = e_attr
                        break
                break
        if experts_attr is not None and router_attr is not None:
            break

    if experts_attr is None or router_attr is None:
        raise ValueError(
            "Could not find a layer with both an experts module and a router "
            "module in the model. Make sure the model has MoE layers, and "
            "that the name of its experts module is in EXPERTS_ATTRS and it "
            "the name of its router module is in ROUTER_ATTRS in "
            "pruning/utils.py"
        )

    # find the MoE layers. Only LinearExperts2D experts are supported, as
    # they receive the router's top-k indices and weights in their forward
    # pass; Llama4LinearExperts and GptOssLinearExperts diverge from this
    # behavior, so they are skipped
    moe_layer_names = []
    for name, module in model.named_modules():
        if not hasattr(module, experts_attr):
            continue

        # check router
        if not hasattr(module, router_attr):
            raise ValueError(f"Could not find router module {name}.{router_attr}")

        if any(re.search(pattern, name) for pattern in ignore):
            continue

        # check experts
        experts: torch.nn.Module = getattr(module, experts_attr)
        if not isinstance(experts, LinearExperts2D):
            logger.warning(
                f"Skipping layer {name}: experts module is not LinearExperts2D"
            )
            continue
        if isinstance(experts, (Llama4LinearExperts, GptOssLinearExperts)):
            logger.warning(f"Skipping unsupported {experts.__class__.__name__}: {name}")
            continue

        moe_layer_names.append(name)

    if not moe_layer_names:
        raise ValueError(
            "Could not find any supported MoE layers with experts in "
            "LinearExperts2D format. Make sure the model has MoE layers "
            "(excluding Llama4LinearExperts and GptOssLinearExperts), and "
            "that the name of its experts module is in EXPERTS_ATTRS and it "
            "the name of its router module is in ROUTER_ATTRS in "
            "pruning/utils.py"
        )

    logger.info(f"Found {len(moe_layer_names)} MoE layers with experts")

    return MoeModelAttrs(
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        top_k_group=top_k_group,
        group_size=group_size,
        experts_attr=experts_attr,
        router_attr=router_attr,
        moe_layer_names=moe_layer_names,
    )


def build_moe_num_experts(
    model: PreTrainedModel,
    retained_experts: dict[str, list[int]],
) -> list[int | None]:
    """
    Build the per-decoder-layer retained-expert-count list for
    :class:`~compressed_tensors.config.ExpertSparsityConfig`.

    The list length equals the number of decoder layers. Dense layers (and MoE
    layers that were not pruned) are ``None``; pruned MoE layers hold their
    retained expert count.
    """
    config = model.config.get_text_config()
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(
            "Could not determine num_hidden_layers to build moe_sparsity_config"
        )

    per_layer: list[int | None] = [None] * num_layers
    for layer_name, retained in retained_experts.items():
        match = re.search(r"layers\.(\d+)", layer_name)
        if match is None:
            raise ValueError(
                f"Could not extract a decoder layer index from '{layer_name}'"
            )
        per_layer[int(match.group(1))] = len(retained)

    return per_layer


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

    # Some models (e.g., Hy3) place e_score_correction_bias on the MoE block
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
