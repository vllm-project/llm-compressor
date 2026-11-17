"""
Dynamic AWQ mapping builders for hybrid attention models.

Models with hybrid attention (mix of full self-attention and linear/Gated
DeltaNet attention) need layer-index-specific AWQ mappings that vary by
model size. This module provides runtime detection and mapping generation
for such architectures (e.g. Qwen3Next, Qwen3.5).
"""

import re
from collections.abc import Callable

from loguru import logger
from torch.nn import Module
from transformers import PreTrainedModel

from llmcompressor.modifiers.transform.awq.mappings import (
    AWQ_MAPPING_REGISTRY,
    AWQMapping,
    default_mappings,
)
from llmcompressor.modifiers.transform.utils.hybrid_attention import (
    get_hybrid_attention_config,
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
    result = get_hybrid_attention_config(model)
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
                    # TODO: should add "re:.*mlp.gate.weight$" but is a Parameter
                    "re:.*mlp.experts.*.gate_proj$",
                    "re:.*mlp.experts.*.up_proj$",
                    "re:.*mlp.shared_expert_gate$",
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


def build_step3p5_mappings(model: Module) -> list[AWQMapping] | None:
    """
    Dynamically build AWQ mappings for Step3p5 models.

    Step-3.5-Flash uses dense FFN layers early in the stack and MoE FFN layers
    later in the stack. The dense and MoE post-attention mappings must be
    layer-index-specific so AWQ only groups each norm with balance layers that
    exist in the same decoder layer.
    """
    dense_indices, moe_indices = _detect_step3p5_ffn_layer_indices(model)

    if not dense_indices and not moe_indices:
        logger.warning(
            "Step3p5 model detected but dense/MoE FFN layer indices could not be "
            "inferred. Falling back."
        )
        return None

    mappings = [
        AWQMapping(
            "re:.*input_layernorm$",
            [
                "re:.*self_attn.q_proj$",
                "re:.*self_attn.k_proj$",
                "re:.*self_attn.v_proj$",
                "re:.*self_attn.g_proj$",
            ],
        ),
        AWQMapping("re:.*self_attn.v_proj$", ["re:.*self_attn.o_proj$"]),
    ]

    if dense_indices:
        dense_re = "|".join(str(i) for i in dense_indices)
        mappings.append(
            AWQMapping(
                f"re:.*layers\\.({dense_re})\\.post_attention_layernorm$",
                [
                    "re:.*mlp.gate_proj$",
                    "re:.*mlp.up_proj$",
                ],
            )
        )

    if moe_indices:
        moe_re = "|".join(str(i) for i in moe_indices)
        mappings.append(
            AWQMapping(
                f"re:.*layers\\.({moe_re})\\.post_attention_layernorm$",
                [
                    "re:.*moe.gate$",
                    "re:.*moe.gate_proj$",
                    "re:.*moe.up_proj$",
                    "re:.*share_expert.gate_proj$",
                    "re:.*share_expert.up_proj$",
                ],
            )
        )

    # The packed moe.up_proj -> moe.down_proj path is intentionally excluded
    # because AWQ's smooth-layer update assumes a 1D/2D smooth weight.
    mappings.append(
        AWQMapping(
            "re:.*(mlp|share_expert).up_proj$",
            ["re:.*(mlp|share_expert).down_proj$"],
        )
    )

    logger.info(
        f"Built dynamic Step3p5 AWQ mappings: "
        f"{len(dense_indices)} dense layers, {len(moe_indices)} MoE layers"
    )

    return mappings


def build_gemma4_mappings(model: PreTrainedModel) -> list[AWQMapping] | None:
    """
    Build AWQ mappings for Gemma4ForCausalLM, Gemma4ForConditionalGeneration,
    and Gemma4UnifiedForConditionalGeneration.

    Gemma4 has hybrid attention and optional KV-cache sharing across the final
    ``num_kv_shared_layers`` decoder layers:
    - non-KV-shared sliding_attention layers: have q_proj, k_proj, and v_proj
    - non-KV-shared full_attention layers: share k=v when ``attention_k_eq_v``
      is True (q_proj and k_proj only; no separate v_proj)
    - KV-shared layers: reuse an earlier layer's KV states and only define
      q_proj (no k_proj or v_proj)

    v_proj -> o_proj is not mapped. Sliding layers use GQA, so the smooth and
    balance widths differ, and full-attention layers with ``attention_k_eq_v``
    reuse ``k_proj`` as ``v`` (changing ``k_proj`` would change attention).
    KV-shared layers map ``input_layernorm`` to ``q_proj`` only.
    """
    config = model.config
    text_config = config.get_text_config()
    layer_types = getattr(text_config, "layer_types", None)
    num_layers = getattr(text_config, "num_hidden_layers", None)

    if layer_types is None or num_layers is None:
        return None

    num_kv_shared = getattr(text_config, "num_kv_shared_layers", 0) or 0
    first_kv_shared = max(0, num_layers - num_kv_shared)
    attention_k_eq_v = bool(getattr(text_config, "attention_k_eq_v", True))

    active_types = layer_types[:num_layers]
    qkv_indices = [
        i
        for i, t in enumerate(active_types[:first_kv_shared])
        if t == "sliding_attention" or (t == "full_attention" and not attention_k_eq_v)
    ]
    qk_indices = [
        i
        for i, t in enumerate(active_types[:first_kv_shared])
        if t == "full_attention" and attention_k_eq_v
    ]
    kv_shared_indices = list(range(first_kv_shared, min(num_layers, len(active_types))))

    if not qkv_indices and not qk_indices and not kv_shared_indices:
        return None

    has_language_model = any(
        name == "language_model" or name.endswith(".language_model")
        for name, _ in model.named_modules()
    )
    prefix = r"re:.*language_model\." if has_language_model else "re:.*"

    mappings = []

    # Non-KV-shared layers with separate q/k/v projections
    if qkv_indices:
        qkv_re = "|".join(str(i) for i in qkv_indices)
        mappings.append(
            AWQMapping(
                f"{prefix}layers\\.({qkv_re})\\.input_layernorm$",
                [
                    f"{prefix}layers\\.({qkv_re})\\.self_attn\\.q_proj$",
                    f"{prefix}layers\\.({qkv_re})\\.self_attn\\.k_proj$",
                    f"{prefix}layers\\.({qkv_re})\\.self_attn\\.v_proj$",
                ],
            )
        )
        # Skip v_proj -> o_proj: GQA makes the smooth and balance widths differ.

    # Non-KV-shared full-attention layers with k=v (no separate v_proj)
    if qk_indices:
        qk_re = "|".join(str(i) for i in qk_indices)
        mappings.append(
            AWQMapping(
                f"{prefix}layers\\.({qk_re})\\.input_layernorm$",
                [
                    f"{prefix}layers\\.({qk_re})\\.self_attn\\.q_proj$",
                    f"{prefix}layers\\.({qk_re})\\.self_attn\\.k_proj$",
                ],
            )
        )
        # No k_proj -> o_proj mapping, altering k_proj would change attention behavior

    # KV-shared layers only have q_proj (no k_proj or v_proj)
    if kv_shared_indices:
        kv_shared_re = "|".join(str(i) for i in kv_shared_indices)
        mappings.append(
            AWQMapping(
                f"{prefix}layers\\.({kv_shared_re})\\.input_layernorm$",
                [
                    f"{prefix}layers\\.({kv_shared_re})\\.self_attn\\.q_proj$",
                ],
            )
        )

    all_re = "|".join(str(i) for i in range(num_layers))
    is_moe = bool(getattr(text_config, "enable_moe_block", False))

    # Dense MLP: pre_feedforward_layernorm -> mlp gate/up_proj
    mappings.append(
        AWQMapping(
            f"{prefix}layers\\.({all_re})\\.pre_feedforward_layernorm$",
            [
                f"{prefix}layers\\.({all_re})\\.mlp\\.gate_proj$",
                f"{prefix}layers\\.({all_re})\\.mlp\\.up_proj$",
            ],
        )
    )

    # MoE experts: pre_feedforward_layernorm_2 -> experts gate/up_proj
    if is_moe:
        mappings.append(
            AWQMapping(
                f"{prefix}layers\\.({all_re})\\.pre_feedforward_layernorm_2$",
                [
                    f"{prefix}layers\\.({all_re})\\.experts\\..*\\.gate_proj$",
                    f"{prefix}layers\\.({all_re})\\.experts\\..*\\.up_proj$",
                ],
            )
        )

    mappings.append(
        AWQMapping(
            f"{prefix}layers\\..*up_proj$",
            [f"{prefix}layers\\..*down_proj$"],
        )
    )

    logger.info(
        f"Built dynamic Gemma4 AWQ mappings: "
        f"{len(qkv_indices)} q/k/v layers, "
        f"{len(qk_indices)} q/k (k=v) layers, "
        f"{len(kv_shared_indices)} kv-shared (q-only) layers, MoE: {is_moe}"
    )

    return mappings


# Builders may require a `PreTrainedModel` even though the public entrypoint
# accepts any `Module`, so the value type is relaxed to `Callable[...]`
AWQ_DYNAMIC_MAPPING_REGISTRY: dict[str, Callable[..., list[AWQMapping] | None]] = {
    "Gemma4ForCausalLM": build_gemma4_mappings,
    "Gemma4ForConditionalGeneration": build_gemma4_mappings,
    "Gemma4UnifiedForConditionalGeneration": build_gemma4_mappings,
    "Qwen3NextForCausalLM": build_hybrid_attention_mappings,
    "Qwen3_5ForCausalLM": build_hybrid_attention_mappings,
    "Qwen3_5ForConditionalGeneration": build_hybrid_attention_mappings,
    "Qwen3_5MoeForCausalLM": build_hybrid_attention_mappings,
    "Qwen3_5MoeForConditionalGeneration": build_hybrid_attention_mappings,
    "Step3p5ForCausalLM": build_step3p5_mappings,
}


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


_STEP3P5_FFN_LAYER_PATTERN = re.compile(
    r"(?:^|\.)layers\.(?P<idx>\d+)\.(?P<ffn>mlp|moe|share_expert)(?:\.|$)"
)


def _detect_step3p5_ffn_layer_indices(model: Module) -> tuple[list[int], list[int]]:
    """
    Detect which Step3p5 decoder layers use dense MLPs and which use MoE blocks.
    """
    dense_indices: set[int] = set()
    moe_indices: set[int] = set()

    for name, _ in model.named_modules():
        match = _STEP3P5_FFN_LAYER_PATTERN.search(name)
        if match is None:
            continue

        layer_idx = int(match.group("idx"))
        ffn_type = match.group("ffn")
        if ffn_type == "mlp":
            dense_indices.add(layer_idx)
        else:
            moe_indices.add(layer_idx)

    return sorted(dense_indices), sorted(moe_indices)
