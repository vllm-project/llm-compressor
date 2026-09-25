from loguru import logger
from pydantic import BaseModel, field_validator
from transformers import PreTrainedModel

__all__ = ["NormMapping", "infer_norm_mapping_from_model"]


class NormMapping(BaseModel):
    """
    SpinQuant needs to know where every norm layer exists in the model,
    as well as all the subsequent Linear layers the norm passes into.
    This is because the norm layer weights need to normalized before
    transforms can be fused into Linear layers.

    :param norm: name or regex that matches norm layer in model
    :param linears: list of names or regexes of Linear layers that
    receive input from norm.
    """

    norm: str
    linears: list[str]

    @field_validator("linears", mode="before")
    def cast_to_list(cls, value):
        if isinstance(value, str):
            return [value]

        return value


_default_mappings = [
    NormMapping(
        norm="re:.*input_layernorm$",
        linears=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    NormMapping(
        norm="re:.*post_attention_layernorm$",
        linears=["re:.*up_proj$", "re:.*gate_proj$"],
    ),
    NormMapping(
        norm="model.norm",
        linears=["lm_head"],
    ),
]

# Cohere2MoE uses a parallel block: a single input_layernorm feeds attention (q/k/v),
# MLP (gate/up) AND the router; these span different parents and the router is absent in
# the dense first layer, so `match_modules_set` can't group them.
# Input_layernorm fusion is handled per-layer in `prepare_cohere2_moe_for_spinquant`;
# only the final norm -> lm_head fusion remains here.
_cohere2_moe_mappings = [
    NormMapping(
        norm="model.norm",
        linears=["lm_head"],
    ),
]

NORM_MAPPING_REGISTRY: dict[str, list[NormMapping]] = {
    "LlamaForCausalLM": _default_mappings,
    "Cohere2MoeForCausalLM": _cohere2_moe_mappings,
}


def infer_norm_mapping_from_model(model: PreTrainedModel) -> list[NormMapping]:
    """
    Infer norm mappings from a model. Checks the dynamic norm mapping registry
    first, then falls back to the static registry, then to defaults.

    :param model: the model to infer norm mappings for
    :return: list of NormMapping for the model
    """
    # Imported lazily to avoid a circular import: dynamic_mappings imports the
    # static registry and NormMapping from this module.
    from llmcompressor.modifiers.transform.spinquant.dynamic_mappings import (
        NORM_DYNAMIC_MAPPING_REGISTRY,
    )

    architecture = model.__class__.__name__

    if architecture in NORM_DYNAMIC_MAPPING_REGISTRY:
        mappings = NORM_DYNAMIC_MAPPING_REGISTRY[architecture](model)
        if mappings is not None:
            return mappings

    if architecture in NORM_MAPPING_REGISTRY:
        return NORM_MAPPING_REGISTRY[architecture]

    logger.info(
        f"Architecture {architecture} not found in norm mappings. "
        f"Using default norm mappings: {_default_mappings}"
    )
    return _default_mappings
