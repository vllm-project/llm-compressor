from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator
from transformers import PreTrainedModel

__all__ = ["SpinQuantMapping", "infer_mapping_from_model"]


class SpinQuantMapping(BaseModel):
    embedding: str

    attn: str
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


_default_mappings = SpinQuantMapping(
    embedding="re:.*embed_tokens$",
    attn="re:.*self_attn$",
    attn_q="re:.*q_proj$",
    attn_k="re:.*k_proj$",
    attn_v="re:.*v_proj$",
    attn_o="re:.*o_proj$",
    mlp_in=["re:.*up_proj$", "re:.*gate_proj$"],
    mlp_out="re:.*down_proj$",
    lm_head="lm_head",
)


SPINQUANT_MAPPING_REGISTRY: Dict[str, SpinQuantMapping] = {
    "LlamaForCausalLM": _default_mappings,
}


def infer_mapping_from_model(model: PreTrainedModel) -> SpinQuantMapping:
    architecture = model.__class__.__name__
    if architecture not in SPINQUANT_MAPPING_REGISTRY:
        logger.info(
            f"Unrecognized model architecture {architecture}. "
            "Falling back to default mappings"
        )

    return SPINQUANT_MAPPING_REGISTRY.get(architecture, _default_mappings)
