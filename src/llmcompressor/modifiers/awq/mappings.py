from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger
from torch.nn import Module

__all__ = ["AWQMapping", "AWQ_MAPPING_REGISTRY", "get_layer_mappings_from_architecture"]


@dataclass
class AWQMapping:
    """
    Dataclass storing config of activation mappings to smooth
    The output activations of smooth_layer are input activations
    into the balance_layers

    `AWQMapping`s are resolved into `ResolvedMapping`s, which
    retain pointers to the actual `torch.nn.Module`s and additional
    metadata at runtime
    """

    smooth_layer: str
    balance_layers: list[str]


_default_mappings = [
    AWQMapping(
        "re:.*input_layernorm",
        ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
    ),
    AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
    AWQMapping(
        "re:.*post_attention_layernorm",
        ["re:.*gate_proj", "re:.*up_proj"],
    ),
    AWQMapping(
        "re:.*up_proj",
        ["re:.*down_proj"],
    ),
]

# Phi merges
#  q, k, and v proj layers into a single qkv_proj layer
#  gate and up proj layers into a single gate_up_proj layer
_phi_mappings = [
    AWQMapping(
        "re:.*input_layernorm",
        ["re:.*qkv_proj"],
    ),
    AWQMapping("re:.*qkv_proj", ["re:.*o_proj"]),
    AWQMapping(
        "re:.*post_attention_layernorm",
        ["re:.*gate_up_proj"],
    ),
    AWQMapping(
        "re:.*gate_up_proj",
        ["re:.*down_proj"],
    ),
]

AWQ_MAPPING_REGISTRY: Dict[str, list[AWQMapping]] = {
    "LlamaForCausalLM": _default_mappings,
    "Qwen2ForCausalLM": _default_mappings,
    "Qwen3ForCausalLM": _default_mappings,
    "MistralForCausalLM": _default_mappings,
    "Phi3ForCausalLM": _phi_mappings,
    "Phi3VForCausalLM": _phi_mappings,
}


@dataclass
class ResolvedMapping:
    """
    Dataclass for storing the resolved mappings between an activation layer
    and the following weights that must be balanced during smoothing

    :param smooth_name: name of the activation layer
    :param smooth_layer: PyTorch module storing the activation layer
    :param balance_layers: list of PyTorch modules that smooth_layer feeds into, must be
        balanced to offset the smoothing of smooth_layer
    :param balance_names: optional list of names of the balance_layers
    :param parent: parent module of the balance_layers
    :param parent_name: name of the parent module
    """

    smooth_name: str
    smooth_layer: Module
    balance_layers: List[Module]
    balance_names: Optional[List[str]] = None
    parent: Optional[Module] = None
    parent_name: Optional[str] = None


def get_layer_mappings_from_architecture(architecture: str) -> List[AWQMapping]:
    """
    :param architecture: str: The architecture of the model
    :return: list: The layer mappings for the given architecture
    """

    if architecture not in AWQ_MAPPING_REGISTRY:
        logger.info(
            f"Architecture {architecture} not found in mappings. "
            f"Using default mappings: {_default_mappings}"
        )

    return AWQ_MAPPING_REGISTRY.get(architecture, _default_mappings)
