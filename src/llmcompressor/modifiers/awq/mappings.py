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
        "re:.*input_layernorm$",
        ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
    AWQMapping(
        "re:.*post_attention_layernorm$",
        ["re:.*gate_proj$", "re:.*up_proj$"],
    ),
    AWQMapping(
        "re:.*up_proj$",
        ["re:.*down_proj$"],
    ),
]

_moe_default_mappings = [
    AWQMapping(
        "re:.*input_layernorm$",
        ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
    AWQMapping(
        "re:.*post_attention_layernorm$",
        ["re:.*mlp.experts.*.gate_proj$", "re:.*mlp.experts.*.up_proj$"],
    ),
    AWQMapping(
        "re:.*up_proj$",
        ["re:.*down_proj$"],
    ),
]

# Phi merges
#  q, k, and v proj layers into a single qkv_proj layer
#  gate and up proj layers into a single gate_up_proj layer
_phi_mappings = [
    AWQMapping(
        "re:.*input_layernorm$",
        ["re:.*qkv_proj$"],
    ),
    AWQMapping("re:.*qkv_proj$", ["re:.*o_proj$"]),
    AWQMapping(
        "re:.*post_attention_layernorm$",
        ["re:.*gate_up_proj$"],
    ),
    AWQMapping(
        "re:.*gate_up_proj$",
        ["re:.*down_proj$"],
    ),
]

# Gemma includes a pre_feedforward_layernorm in between
#  post_attention_layernorm and the mlp down/gate proj layers
#  use that instead of post_attention_layernorm in 3rd mapping:
_gemma_mappings = [
    AWQMapping(
        "re:.*input_layernorm$",
        ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
    AWQMapping(
        "re:.*pre_feedforward_layernorm$",
        ["re:.*gate_proj$", "re:.*up_proj$"],
    ),
    AWQMapping(
        "re:.*up_proj$",
        ["re:.*down_proj$"],
    ),
]


# Cohere architecture is similar to default, with a very fundamental difference.
# The MLP block is executed in parallel to the attention. So the tensor goes
# through input_layernorm and then from there it goes directly to the attention
# module and to the MLP module.
_cohere_mappings = [
    AWQMapping(
        "re:.*input_layernorm$",
        [
            "re:.*self_attn.q_proj$",
            "re:.*self_attn.k_proj$",
            "re:.*self_attn.v_proj$",
            "re:.*mlp.gate_proj$",
            "re:.*mlp.up_proj$",
        ],
    ),
    AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
    AWQMapping(
        "re:.*up_proj$",
        ["re:.*down_proj$"],
    ),
]

# DeepseekV3
_deepseek_mappings = [
    AWQMapping(
        "re:.*input_layernorm$",
        # Some models use q_proj instead of q_a_proj
        ["re:.*(q|q_a)_proj$", "re:.*kv_a_proj_with_mqa$"],
    ),
    AWQMapping("re:.*q_a_layernorm$", ["re:.*q_b_proj$"]),
    AWQMapping("re:.*kv_a_layernorm$", ["re:.*kv_b_proj$"]),
    AWQMapping(
        "re:.*post_attention_layernorm$",
        ["re:.*gate_proj$", "re:.*up_proj$"],
    ),
    AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
]

_bloom_mappings = [
    AWQMapping("re:.*input_layernorm$", ["re:.*query_key_value$"]),
    AWQMapping("re:.*post_attention_layernorm$", ["re:.*dense_h_to_4h$"]),
    AWQMapping("re:.*gelu_impl$", ["re:.*dense_4h_to_h$"]),
    # Note: AutoAWQ excludes this mapping, based on researcher's post in
    # https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
    # AWQMapping(
    #     "re:.*query_key_value$",
    #     ["re:.*dense$"]
    # ),
]
AWQ_MAPPING_REGISTRY: Dict[str, list[AWQMapping]] = {
    "BloomForCausalLM": _bloom_mappings,
    "CohereForCausalLM": _cohere_mappings,
    "Cohere2ForCausalLM": _cohere_mappings,
    "DeepseekV3ForCausalLM": _deepseek_mappings,
    "Gemma2ForCausalLM": _gemma_mappings,
    "Gemma3ForCausalLM": _gemma_mappings,
    "Gemma3ForConditionalGeneration": _gemma_mappings,
    "LlamaForCausalLM": _default_mappings,
    "Mistral3ForConditionalGeneration": _default_mappings,
    "MistralForCausalLM": _default_mappings,
    "Phi3ForCausalLM": _phi_mappings,
    "Phi3VForCausalLM": _phi_mappings,
    "Qwen2ForCausalLM": _default_mappings,
    "Qwen2_5OmniThinkerForConditionalGeneration": _default_mappings,
    "Qwen2MoeForCausalLM": _moe_default_mappings,
    "Qwen3ForCausalLM": _default_mappings,
    "Qwen3MoeForCausalLM": _moe_default_mappings,
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
