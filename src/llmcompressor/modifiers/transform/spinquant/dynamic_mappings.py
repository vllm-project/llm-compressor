"""
Dynamic SpinQuant mapping builders for hybrid attention models.

Models with hybrid attention (a mix of full self-attention and linear/Gated
DeltaNet attention) need mappings that account for the fact that only
full-attention layers expose ``q/k/v/o_proj`` projections, while
linear-attention layers expose ``linear_attn.in_proj_*`` / ``out_proj``
projections instead. This module provides runtime detection and mapping
generation for such architectures (e.g. Qwen3.5).

R1 rotates the residual stream: the linear-attention input projections
(``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a``) read the rotated
residual stream and must be inversely rotated alongside ``q/k/v_proj``, while
``linear_attn.out_proj`` writes the residual stream and must be rotated
alongside ``o_proj``. Both are folded into ``mlp_in`` / ``mlp_out``, which R1
alone consumes, so they are safe to share with the MLP projections.

.. note::
    Only R1 is supported for Qwen3.5. R2 rotates the value/output head space,
    but Qwen3.5's full-attention layers apply an element-wise gate
    (``attn_output * sigmoid(gate)``) between the attention output and
    ``o_proj`` inside that same head space, which does not commute with R2 and
    breaks its logits invariance. R3/R4 are online rotations and are unrelated
    to these mappings.
"""

from collections.abc import Callable

from loguru import logger
from torch.nn import Module

from llmcompressor.modifiers.transform.spinquant.mappings import SpinQuantMapping
from llmcompressor.modifiers.transform.spinquant.norm_mappings import NormMapping
from llmcompressor.modifiers.transform.utils.hybrid_attention import (
    detect_linear_attn_projections,
    get_hybrid_attention_config,
)

__all__ = [
    "SPINQUANT_DYNAMIC_MAPPING_REGISTRY",
    "NORM_DYNAMIC_MAPPING_REGISTRY",
]


def build_qwen3_5_spinquant_mapping(model: Module) -> SpinQuantMapping | None:
    """
    Dynamically build a SpinQuantMapping for dense Qwen3.5 hybrid-attention
    models.

    Linear-attention input projections are folded into ``mlp_in`` and
    ``linear_attn.out_proj`` into ``mlp_out`` so R1 rotates the residual stream
    consistently across both full- and linear-attention layers. Returns None if
    the model is not a dense hybrid-attention model.
    """
    if get_hybrid_attention_config(model) is None:
        return None

    linear_proj_names = detect_linear_attn_projections(model)

    mlp_in = ["re:.*up_proj$", "re:.*gate_proj$"]
    mlp_out = ["re:.*down_proj$"]
    if linear_proj_names:
        mlp_in += [rf"re:.*linear_attn\.{p}$" for p in linear_proj_names]
        mlp_out.append(r"re:.*linear_attn\.out_proj$")

    return SpinQuantMapping(
        embedding="re:.*embed_tokens$",
        attn="re:.*self_attn$",
        attn_q="re:.*q_proj$",
        attn_k="re:.*k_proj$",
        attn_v="re:.*v_proj$",
        attn_o="re:.*o_proj$",
        mlp_in=mlp_in,
        mlp_out=mlp_out,
        lm_head="lm_head",
    )


def build_qwen3_5_norm_mappings(model: Module) -> list[NormMapping] | None:
    """
    Dynamically build norm mappings for dense Qwen3.5 hybrid-attention models.

    ``input_layernorm`` feeds ``q/k/v_proj`` in full-attention layers and
    ``linear_attn.in_proj_*`` in linear-attention layers, so its regex must be
    restricted to the corresponding layer indices for ``match_modules_set`` to
    group one norm per layer. Returns None if the model is not a dense
    hybrid-attention model.
    """
    result = get_hybrid_attention_config(model)
    if result is None:
        return None

    layer_types, _ = result
    full_indices = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    linear_indices = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    if not full_indices or not linear_indices:
        return None

    full_re = "|".join(str(i) for i in full_indices)
    linear_re = "|".join(str(i) for i in linear_indices)
    linear_proj_names = detect_linear_attn_projections(model)

    mappings = [
        NormMapping(
            norm=rf"re:.*layers\.({full_re})\.input_layernorm$",
            linears=[
                r"re:.*self_attn\.q_proj$",
                r"re:.*self_attn\.k_proj$",
                r"re:.*self_attn\.v_proj$",
            ],
        ),
    ]

    if linear_proj_names:
        mappings.append(
            NormMapping(
                norm=rf"re:.*layers\.({linear_re})\.input_layernorm$",
                linears=[rf"re:.*linear_attn\.{p}$" for p in linear_proj_names],
            )
        )

    mappings.append(
        NormMapping(
            norm="re:.*post_attention_layernorm$",
            linears=["re:.*up_proj$", "re:.*gate_proj$"],
        )
    )

    final_norm = _detect_final_norm_name(model)
    if final_norm is not None:
        mappings.append(NormMapping(norm=final_norm, linears=["lm_head"]))

    logger.info(
        f"Built dynamic Qwen3.5 SpinQuant norm mappings: "
        f"{len(full_indices)} full-attention layers, "
        f"{len(linear_indices)} linear-attention layers, "
        f"linear projections: {linear_proj_names}"
    )

    return mappings


def _detect_final_norm_name(model: Module) -> str | None:
    """
    Detect the final residual-stream norm module name (the one feeding lm_head).

    The final norm is a direct child of the language/text model, so it is not
    nested under a decoder layer and does not belong to the visual tower. We
    locate the decoder-layers prefix first and resolve its sibling ``norm``,
    falling back to a name-based heuristic if no layers module is found.
    """
    layers_prefix = None
    for name, _ in model.named_modules():
        if name == "layers" or name.endswith(".layers"):
            layers_prefix = name
            break

    if layers_prefix is not None:
        parent = layers_prefix.rsplit(".", 1)[0] if "." in layers_prefix else ""
        expected_norm = f"{parent}.norm" if parent else "norm"
        for name, _ in model.named_modules():
            if name == expected_norm:
                return name

    for name, _ in model.named_modules():
        if not name.endswith(".norm"):
            continue
        if ".layers." in name or "visual" in name:
            continue
        return name
    return None


SPINQUANT_DYNAMIC_MAPPING_REGISTRY: dict[
    str, Callable[[Module], SpinQuantMapping | None]
] = {
    "Qwen3_5ForCausalLM": build_qwen3_5_spinquant_mapping,
    "Qwen3_5ForConditionalGeneration": build_qwen3_5_spinquant_mapping,
}

NORM_DYNAMIC_MAPPING_REGISTRY: dict[
    str, Callable[[Module], list[NormMapping] | None]
] = {
    "Qwen3_5ForCausalLM": build_qwen3_5_norm_mappings,
    "Qwen3_5ForConditionalGeneration": build_qwen3_5_norm_mappings,
}
