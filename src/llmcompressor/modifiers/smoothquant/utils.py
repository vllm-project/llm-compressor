from collections import namedtuple
from typing import Dict, List, Tuple, Union

from loguru import logger

__all__ = [
    "get_layer_mappings_from_architecture",
    "MAPPINGS_REGISTRY",
    "DEFAULT_SMOOTHQUANT_MAPPINGS",
]

LayerMapType = Tuple[Union[List[str], str], Union[List[str], str]]
LayerMap: LayerMapType = namedtuple("LayerMap", ["balance_layers", "smooth_layers"])

LLAMA_MAPPINGS: List[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*gate_proj", "re:.*up_proj"],
        smooth_layers="re:.*post_attention_layernorm",
    ),
]
MIXTRAL_MAPPINGS: List[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*gate"], smooth_layers="re:.*post_attention_layernorm"
    ),
]

# Registry of layer mappings for different architectures
#   Add more mappings here
MAPPINGS_REGISTRY: Dict[str, List[LayerMap]] = {
    "LlamaForCausalLM": LLAMA_MAPPINGS,
    "MixtralForCausalLM": MIXTRAL_MAPPINGS,
}

# Default mappings to use if architecture is not found in the registry
DEFAULT_SMOOTHQUANT_MAPPINGS: List[LayerMap] = LLAMA_MAPPINGS


def get_layer_mappings_from_architecture(architecture: str) -> List[LayerMap]:
    """
    :param architecture: str: The architecture of the model
    :return: list: The layer mappings for the given architecture
    """

    if architecture not in MAPPINGS_REGISTRY:
        logger.info(
            f"Architecture {architecture} not found in mappings. "
            f"Using default mappings: {DEFAULT_SMOOTHQUANT_MAPPINGS}"
        )

    return MAPPINGS_REGISTRY.get(architecture, DEFAULT_SMOOTHQUANT_MAPPINGS)
