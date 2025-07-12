from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SpinQuantMappings(BaseModel):
    embedding: str

    attn_q: str
    attn_k: str
    attn_v: str
    attn_o: str
    attn_head_dim: Optional[int] = Field(default=None)

    mlp_in: List[str]  # up_proj, gate_proj
    mlp_out: List[str]  # down_proj

    lm_head: str

    @field_validator("mlp_in", "mlp_out", mode="before")
    def cast_to_list(cls, value):
        if isinstance(value, str):
            return [value]

        return value


_default_mappings = SpinQuantMappings(
    embedding="re:.*embed_tokens$",
    attn_q="re:.*q_proj$",
    attn_k="re:.*k_proj$",
    attn_v="re:.*v_proj$",
    attn_o="re:.*o_proj$",
    mlp_in=["re:.*up_proj$", "re:.*gate_proj$"],
    mlp_out="re:.*down_proj$",
    lm_head="lm_head",
)


SPINQUANT_MAPPING_REGISTRY: Dict[str, SpinQuantMappings] = {
    "LlamaForCausalLM": _default_mappings,
}
