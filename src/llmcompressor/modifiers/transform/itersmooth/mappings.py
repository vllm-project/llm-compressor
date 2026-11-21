from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SubgraphType = Literal["norm-linear", "linear-linear", "ov", "up-down"]


class IterSmoothMapping(BaseModel):
    """A fusible source and all consumers that must compensate its scaling.

    Names support the same ``re:`` patterns as SmoothQuant. ``fused`` selects
    the last V block of a contiguous [Q, K, V] projection, or the second half
    of a contiguous [gate, up] projection. Interleaved layouts are unsupported.
    Head counts can be specified for OV or inferred from the model config.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    targets: list[str] = Field(min_length=1)
    subgraph_type: SubgraphType = "norm-linear"
    num_attention_heads: int | None = Field(default=None, gt=0)
    num_key_value_heads: int | None = Field(default=None, gt=0)
    fused: bool = False
