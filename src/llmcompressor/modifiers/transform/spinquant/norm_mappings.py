from typing import Callable, Dict, List, Union

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

def _build_deepseek_v3_mappings(model: PreTrainedModel) -> List[NormMapping]:
    k = getattr(model.config, "first_k_dense_replace", 1)
    # regex matching dense layers: 0, 1, ..., k-1
    dense_re = "|".join(str(i) for i in range(k))
    # regex matching MoE layers: k, k+1, ...
    # matches any number that is NOT one of the dense layer indices
    moe_re = rf"(?!({'|'.join(str(i) for i in range(k))})(?:\D|$))\d+"

    mappings = [
        NormMapping(
            norm="re:.*input_layernorm$",
            linears=["re:.*q(_a)?_proj$", "re:.*kv_a_proj_with_mqa$"],
        ),
        # dense layers
        NormMapping(
            norm=rf"re:.*\.({dense_re})\.post_attention_layernorm$",
            linears=[
                rf"re:.*\.({dense_re})\.mlp\.up_proj$",
                rf"re:.*\.({dense_re})\.mlp\.gate_proj$",
            ],
        ),
        # MoE layers
        NormMapping(
            norm=rf"re:.*\.{moe_re}\.post_attention_layernorm$",
            linears=[
                rf"re:.*\.{moe_re}\.mlp\.shared_experts\.up_proj$",
                rf"re:.*\.{moe_re}\.mlp\.shared_experts\.gate_proj$",
                rf"re:.*\.{moe_re}\.mlp\.gate$",
                r"re:.*\.mlp\.experts\.\d+\.up_proj$",
                r"re:.*\.mlp\.experts\.\d+\.gate_proj$",
            ],
        ),
        NormMapping(
            norm="model.norm",
            linears=["lm_head"],
        ),
    ]
    return mappings


NORM_MAPPING_REGISTRY: Dict[
    str, Union[List[NormMapping], Callable[[PreTrainedModel], List[NormMapping]]]
] = {
    "LlamaForCausalLM": _default_mappings,
    "DeepseekV3ForCausalLM": _build_deepseek_v3_mappings,
}


def infer_norm_mapping_from_model(model: PreTrainedModel) -> List[NormMapping]:
    architecture = model.__class__.__name__
    if architecture not in NORM_MAPPING_REGISTRY:
        logger.info(
            f"Unrecognized model architecture {architecture}. "
            "Falling back to default mappings"
        )

    entry = NORM_MAPPING_REGISTRY.get(architecture, _default_mappings)
    if callable(entry):
        return entry(model)
    return entry
