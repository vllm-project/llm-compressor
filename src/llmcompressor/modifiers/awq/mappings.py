from dataclasses import dataclass

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

    :param smooth_layer: regex or name of the activation layer to smooth
    :param balance_layers: list of regex or names of weight layers that must
        be balanced to offset the smoothing
    :param activation_hook_target: optional dotted attribute path relative to the
        parent module (lowest common ancestor of balance_layers) specifying which
        submodule to hook for activation caching. Useful for parallel transformer
        blocks (e.g. Cohere, Gemma 3) where the first balance layer is not the
        correct place to capture activations. When ``None`` (default), the hook
        is placed on ``balance_layers[0]``.
    """

    smooth_layer: str
    balance_layers: list[str]
    activation_hook_target: str | None = None


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

# Qwen3Next uses hybrid attention: self_attn (layers 3,7,11,...) and
# linear_attn/Gated DeltaNet (all other layers). Layer-specific patterns
# are required since different layers have different projection structures.
# Also includes shared_expert in the MoE MLP.
# TODO: The self_attn layer indices are hardcoded for the 80B variant (48 layers,
# interval=4, starting at layer 3). The interval is a configurable parameter in the
# model config (full_attention_layer_interval). Consider making this dynamic.
_qwen3_next_moe_mappings = [
    AWQMapping(
        "re:.*layers\\.(3|7|11|15|19|23|27|31|35|39|43|47)\\.input_layernorm$",
        ["re:.*self_attn.q_proj$", "re:.*self_attn.k_proj$", "re:.*self_attn.v_proj$"],
    ),
    AWQMapping("re:.*self_attn.v_proj$", ["re:.*self_attn.o_proj$"]),
    AWQMapping(
        "re:.*layers\\.(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30|32|33|34|36|37|38|40|41|42|44|45|46)\\.input_layernorm$",
        ["re:.*linear_attn.in_proj_qkvz$", "re:.*linear_attn.in_proj_ba$"],
    ),
    AWQMapping(
        "re:.*post_attention_layernorm$",
        [
            "re:.*mlp.experts.*.gate_proj$",
            "re:.*mlp.experts.*.up_proj$",
            "re:.*mlp.shared_expert.gate_proj$",
            "re:.*mlp.shared_expert.up_proj$",
        ],
    ),
    AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
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

# Exaone4
_exaone4_mappings = [
    AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
    AWQMapping(
        "re:.*up_proj$",
        ["re:.*down_proj$"],
    ),
]

# AFMOE uses dual normalization: pre_mlp_layernorm feeds the MLP
# (not post_attention_layernorm) and attention has its own gate_proj
# for gating mechanism
_afmoe_mappings = [
    AWQMapping(
        "re:.*input_layernorm$",
        [
            "re:.*self_attn.q_proj$",
            "re:.*self_attn.k_proj$",
            "re:.*self_attn.v_proj$",
            "re:.*self_attn.gate_proj$",
        ],
    ),
    AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
    AWQMapping(
        "re:.*pre_mlp_layernorm$",
        ["re:.*mlp.*gate_proj$", "re:.*mlp.*up_proj$"],
    ),
    AWQMapping(
        "re:.*up_proj$",
        ["re:.*down_proj$"],
    ),
]

# Example mapping for MoE models with parallel transformer blocks, where
# attention and MoE share the same input. This is the only case where
# activation_hook_target is needed. Without it, the hook lands on
# balance_layers[0] — which could be a single expert — capturing only that expert's
# input rather than the full activation flowing into the MLP & Attention branch.
# Setting activation_hook_target="mlp" hooks parent.mlp instead, so the cached
# activations reflect the complete input to the MoE & Attention branch.
_example_parallel_transformer_block_mappings = [
    AWQMapping(
        "re:.*input_layernorm$",
        [
            "re:.*mlp.experts.[0-9]+.gate_proj$",
            "re:.*mlp.experts.[0-9]+.up_proj$",
            "re:.*mlp.shared_experts.gate_proj$",
            "re:.*mlp.shared_experts.up_proj$",
            "re:.*mlp.gate$",
            "re:.*q_proj$",
            "re:.*k_proj$",
            "re:.*v_proj$",
        ],
        activation_hook_target="mlp",
    )
]

AWQ_MAPPING_REGISTRY: dict[str, list[AWQMapping]] = {
    "AfmoeForCausalLM": _afmoe_mappings,
    "BloomForCausalLM": _bloom_mappings,
    "CohereForCausalLM": _cohere_mappings,
    "Cohere2ForCausalLM": _cohere_mappings,
    "Cohere2VisionForConditionalGeneration": _cohere_mappings,
    "DeepseekV3ForCausalLM": _deepseek_mappings,
    "Exaone4ForCausalLM": _exaone4_mappings,
    "Gemma2ForCausalLM": _gemma_mappings,
    "Gemma3ForCausalLM": _gemma_mappings,
    "Gemma3ForConditionalGeneration": _gemma_mappings,
    "LlamaForCausalLM": _default_mappings,
    "Llama4ForConditionalGeneration": _default_mappings,
    "Mistral3ForConditionalGeneration": _default_mappings,
    "MistralForCausalLM": _default_mappings,
    "Olmo3ForCausalLM": _exaone4_mappings,
    "Phi3ForCausalLM": _phi_mappings,
    "Phi3VForCausalLM": _phi_mappings,
    "Qwen2ForCausalLM": _default_mappings,
    "Qwen2_5OmniThinkerForConditionalGeneration": _default_mappings,
    "Qwen2MoeForCausalLM": _moe_default_mappings,
    "Qwen3ForCausalLM": _default_mappings,
    "Qwen3MoeForCausalLM": _moe_default_mappings,
    "Qwen3NextForCausalLM": _qwen3_next_moe_mappings,
    "Glm4MoeForCausalLM": _default_mappings,
    "SeedOssForCausalLM": _default_mappings,
    "Ernie4_5_MoeForCausalLM": _default_mappings,
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
    :param activation_hook_target: optional resolved module to hook for activation
        caching. When set, the activation cache hook is placed on this module
        instead of ``balance_layers[0]``. Populated from
        ``AWQMapping.activation_hook_target``.
    """

    smooth_name: str
    smooth_layer: Module
    balance_layers: list[Module]
    balance_names: list[str]
    parent: Module
    parent_name: str
    activation_hook_target: Module | None = None


def get_layer_mappings_from_architecture(architecture: str) -> list[AWQMapping]:
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
