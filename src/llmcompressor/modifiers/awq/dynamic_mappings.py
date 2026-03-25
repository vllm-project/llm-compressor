"""
Dynamic AWQ mapping builders for hybrid attention models.

Models with hybrid attention (mix of full self-attention and linear/Gated
DeltaNet attention) need layer-index-specific AWQ mappings that vary by
model size. This module provides runtime detection and mapping generation
for such architectures (e.g. Qwen3Next, Qwen3.5).
"""

from collections.abc import Callable

from loguru import logger
from torch.nn import Module

from llmcompressor.modifiers.awq.mappings import (
    AWQ_MAPPING_REGISTRY,
    AWQMapping,
    default_mappings,
)
from llmcompressor.modifiers.utils.pytorch_helpers import is_moe_model

__all__ = ["AWQ_DYNAMIC_MAPPING_REGISTRY", "get_layer_mappings_from_model"]


def get_layer_mappings_from_model(model: Module) -> list[AWQMapping]:
    """
    Infer AWQ mappings from a model. Checks the dynamic mapping registry
    first (for models needing runtime-generated mappings), then falls back
    to the static registry, then to default mappings.

    :param model: the model to infer mappings for
    :return: list of AWQMapping for the model
    """
    model_name = model.__class__.__name__

    if model_name in AWQ_DYNAMIC_MAPPING_REGISTRY:
        mappings = AWQ_DYNAMIC_MAPPING_REGISTRY[model_name](model)
        if mappings is not None:
            return mappings

    if model_name in AWQ_MAPPING_REGISTRY:
        return AWQ_MAPPING_REGISTRY[model_name]

    logger.info(
        f"Architecture {model_name} not found in mappings. "
        f"Using default mappings: {default_mappings}"
    )
    return default_mappings


def build_hybrid_attention_mappings(model: Module) -> list[AWQMapping] | None:
    """
    Dynamically build AWQ mappings for models with hybrid attention
    (full self-attention + linear/Gated DeltaNet attention), such as
    Qwen3Next and Qwen3.5.

    Reads layer_types from the model config to determine which layers use
    full vs linear attention, then inspects the model's module names to
    detect the correct linear attention projection names and MLP structure.

    Returns None if the model is not a hybrid attention model.
    """
    result = _get_hybrid_attention_config(model)
    if result is None:
        return None

    layer_types, num_layers = result

    full_indices = [i for i in range(num_layers) if layer_types[i] == "full_attention"]
    linear_indices = [
        i for i in range(num_layers) if layer_types[i] == "linear_attention"
    ]

    if not full_indices or not linear_indices:
        logger.warning(
            "Hybrid attention model detected but could not find indices for "
            "both full and linear attention layers. Falling back."
        )
        return None

    full_re = "|".join(str(i) for i in full_indices)
    linear_re = "|".join(str(i) for i in linear_indices)

    linear_proj_names = _detect_linear_attn_projections(model)
    is_moe = is_moe_model(model)

    mappings = []

    # Full attention layers: input_layernorm -> q/k/v_proj
    mappings.append(
        AWQMapping(
            f"re:.*layers\\.({full_re})\\.input_layernorm$",
            [
                "re:.*self_attn.q_proj$",
                "re:.*self_attn.k_proj$",
                "re:.*self_attn.v_proj$",
            ],
        )
    )

    # Linear attention layers: input_layernorm -> linear_attn projections
    if linear_proj_names:
        mappings.append(
            AWQMapping(
                f"re:.*layers\\.({linear_re})\\.input_layernorm$",
                [f"re:.*linear_attn.{p}$" for p in linear_proj_names],
            )
        )

    # MLP mappings depend on whether the model uses MoE
    if is_moe:
        mappings.append(
            AWQMapping(
                "re:.*post_attention_layernorm$",
                [
                    "re:.*mlp.experts.*.gate_proj$",
                    "re:.*mlp.experts.*.up_proj$",
                    "re:.*mlp.shared_expert.gate_proj$",
                    "re:.*mlp.shared_expert.up_proj$",
                ],
            )
        )
    else:
        mappings.append(
            AWQMapping(
                "re:.*post_attention_layernorm$",
                ["re:.*gate_proj$", "re:.*up_proj$"],
            )
        )

    mappings.append(AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]))

    logger.info(
        f"Built dynamic hybrid attention AWQ mappings: "
        f"{len(full_indices)} full-attention layers, "
        f"{len(linear_indices)} linear-attention layers, "
        f"linear projections: {linear_proj_names}, MoE: {is_moe}"
    )

    return mappings


AWQ_DYNAMIC_MAPPING_REGISTRY: dict[str, Callable[[Module], list[AWQMapping] | None]] = {
    "Qwen3NextForCausalLM": build_hybrid_attention_mappings,
    "Qwen3_5ForCausalLM": build_hybrid_attention_mappings,
    "Qwen3_5ForConditionalGeneration": build_hybrid_attention_mappings,
    "Qwen3_5MoeForCausalLM": build_hybrid_attention_mappings,
    "Qwen3_5MoeForConditionalGeneration": build_hybrid_attention_mappings,
}


def _get_hybrid_attention_config(model: Module) -> tuple[list[str], int] | None:
    """
    Extract layer_types and num_hidden_layers from a model with hybrid attention
    (mix of full self-attention and linear/Gated DeltaNet attention).

    Checks both top-level config and text_config (for VL models like Qwen3.5).
    Returns (layer_types, num_hidden_layers) or None if not a hybrid model.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None

    # VL models nest text config under text_config
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)
    num_layers = getattr(text_config, "num_hidden_layers", None)

    if layer_types is None or num_layers is None:
        return None

    has_full = "full_attention" in layer_types
    has_linear = "linear_attention" in layer_types
    if not (has_full and has_linear):
        return None

    return layer_types, num_layers


def _detect_linear_attn_projections(model: Module) -> list[str]:
    """
    Detect the linear attention projection names by inspecting the first
    linear_attention layer's submodules.

    Different architectures use different projection layouts:
      - Qwen3Next: in_proj_qkvz, in_proj_ba
      - Qwen3.5:   in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
    """
    proj_names = []
    for name, _ in model.named_modules():
        if ".linear_attn." not in name:
            continue
        # Extract the submodule name after linear_attn.
        sub = name.rsplit("linear_attn.", 1)[-1]
        # Only include input projection layers (in_proj_*)
        if sub.startswith("in_proj_"):
            proj_names.append(sub)
    # Deduplicate while preserving order (same projections repeat per layer)
    return list(dict.fromkeys(proj_names))
