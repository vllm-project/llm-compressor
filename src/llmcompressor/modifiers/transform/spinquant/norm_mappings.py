from loguru import logger
from pydantic import BaseModel, field_validator
from transformers import PreTrainedModel

__all__ = ["infer_norm_mapping_from_model"]


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
    architecture = model.__class__.__name__
    if architecture not in NORM_MAPPING_REGISTRY:
        logger.info(
            f"Unrecognized model architecture {architecture}. "
            "Falling back to default mappings"
        )

    return NORM_MAPPING_REGISTRY.get(architecture, _default_mappings)
