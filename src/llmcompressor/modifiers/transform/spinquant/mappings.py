from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator
from transformers import PreTrainedModel

__all__ = ["SpinQuantMapping", "infer_mapping_from_model"]


class SpinQuantMapping(BaseModel):
    """
    SpinQuant needs to know the entire architecture of the model,
    as R1, R2, R3, and R4 rotations need to be applied to specific
    layers (https://arxiv.org/pdf/2405.16406 Fig. 1).

    :param embedding: name or regex of embedding layer
    :param attn_q: name or regex of q_proj layer in attention block
    :param attn_k: name or regex of k_proj layer in attention block
    :param attn_v: name or regex of v_proj layer in attention block
    :param attn_o: name or regex of o_proj layer in attention block
    :param attn_head_dim: head_dim of the attention module, needed
        because R2 needs to be applied "head-wisely" to v_proj and
        o_proj
    :param mlp_in: list of names or regexes for the mlp blocks that
        receive the input to the MLP block, usually up_proj and gate_proj
    :param mlp_out: list of names or regexes for the mlp blocks that
        consitute the output of the MLP block, usually down_proj
    """

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


_default_mappings = SpinQuantMapping(
    embedding="re:.*embed_tokens$",
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
