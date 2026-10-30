"""
Utilities shared by the expert pruning modifiers: MoE layer detection.

"""

import re
from dataclasses import dataclass

import torch
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.modeling.moe.gpt_oss import GptOssLinearExperts
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.llama4 import Llama4LinearExperts
from llmcompressor.utils.helpers import hasattr_fallbacks, getattr_fallbacks

__all__ = [
    "MoeModelAttrs",
    "NUM_EXPERTS_CONFIG_KEYS",
    "NUM_EXPERTS_MODULE_KEYS",
    "build_moe_num_experts",
    "get_moe_attrs",
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
    num_layers: int
    num_experts: int
    top_k: int

    # group-limited moe config
    n_group: int | None
    top_k_group: int | None
    group_size: int | None

    # attributes used to find MoE layers
    experts_attr: str
    router_attr: str

    # mapping from layer index to MoE
    # layers without MoEs are None
    moe_layer_names: list[None | str]


def get_moe_attrs(model: PreTrainedModel) -> MoeModelAttrs:
    config = model.config.get_text_config()

    num_experts_config_key = None
    router_attr = None
    experts_attr = None
    moe_layer_names = []

    num_experts = getattr_fallbacks(config, NUM_EXPERTS_CONFIG_KEYS)
    top_k = getattr_fallbacks(config, TOP_K_CONFIG_KEYS)
    n_group = getattr_fallbacks(config, N_GROUP_CONFIG_KEYS)
    top_k_group = getattr_fallbacks(config, TOP_K_GROUP_CONFIG_KEYS)

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
    group_size = num_experts // n_group if n_group is not None else None

    # find the expert and router attribute names using the first match
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
    else:
        raise ValueError(
            "Could not find a layer with both an experts module and a router "
            "module in the model. Make sure the model has MoE layers, and "
            "that the name of its experts module is in EXPERTS_ATTRS and it "
            "the name of its router module is in ROUTER_ATTRS in "
            "pruning/utils.py"
        )

    # find moe layers using expert and router names
    for name, module in model.named_modules():
        if re.search(r"layers\.(\d+)", name) is None:
            
        

        if not hasattr(module, experts_attr):
            continue

        # check router
        if not hasattr(module, router_attr):
            raise ValueError(f"Could not find router module {name}.{router_attr}")
            
        # check experts
        experts: torch.nn.Module = getattr(module, experts_attr)
        if experts.__class__ != LinearExperts2D:
            raise ValueError(f"Unsupported experts type {experts.__class__}")
        
        # record moe names
        moe_layer_names.append(name)

    logger.info(f"Found {len(moe_layer_names)} MoE layers with experts")

    return MoeModelAttrs(
        num_experts_config_key=num_experts_config_key,
        router_attr=router_attr,
        experts_attr=experts_attr,
        moe_layer_names=moe_layer_names,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        top_k_group=top_k_group,
        group_size=group_size,
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
