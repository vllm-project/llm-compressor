from dataclasses import dataclass
from typing import Dict, List, Optional

from torch.nn import Module

__all__ = ["AWQMapping", "AWQ_MAPPING_REGISTRY"]


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


AWQ_MAPPING_REGISTRY: Dict[str, list[AWQMapping]] = {
    "Llama": [
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
    ],
    # TODO (Brian INFERENG-529) Add Qwen mappings
    # "Qwen": [ ],
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
