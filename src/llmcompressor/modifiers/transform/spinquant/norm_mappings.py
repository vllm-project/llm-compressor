from typing import Dict, List

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
    linears: List[str]

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

NORM_MAPPING_REGISTRY: Dict[str, NormMapping] = {
    "LlamaForCausalLM": _default_mappings,
}


def infer_norm_mapping_from_model(model: PreTrainedModel) -> List[NormMapping]:
    architecture = model.__class__.__name__
    if architecture not in NORM_MAPPING_REGISTRY:
        logger.info(
            f"Unrecognized model architecture {architecture}. "
            "Falling back to default mappings"
        )

    return NORM_MAPPING_REGISTRY.get(architecture, _default_mappings)
