from typing import Dict, List

from pydantic import BaseModel, field_validator


class NormMapping(BaseModel):
    norm: str
    linears: List[str]

    @field_validator("linears", mode="before")
    def cast_to_list(cls, value):
        if isinstance(value, str):
            return [value]

        return value


_default_norm_mappings = [
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
    "LlamaForCausalLM": _default_norm_mappings,
}
