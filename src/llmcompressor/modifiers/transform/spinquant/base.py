from typing import Optional, List, Literal

from compressed_tensors.transform import TransformConfig, TransformScheme, TransformArgs, apply_transform_config
from pydantic import BaseModel, field_validator, Field

from llmcompressor.core import Event, EventType, State
from llmcompressor.modeling import fuse_norm_linears
from llmcompressor.modifiers import Modifier
from enum import Enum

from transformers import PreTrainedModel


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
    
class NormMapping(BaseModel):
    norm: str
    linears: List[str]

    @field_validator("linears", mode="before")
    def cast_to_list(cls, value):
        if isinstance(value, str):
            return [value]
        
        return value



llama_spinquant = SpinQuantMappings(
    embedding="re:.*embed_tokens$",

    attn_q="re:.*q_proj$",
    attn_k="re:.*k_proj$",
    attn_v="re:.*v_proj$",
    attn_o="re:.*o_proj$",

    mlp_in=["re:.*up_proj$", "re:.*gate_proj$"],
    mlp_out="re:.*down_proj$",

    lm_head="lm_head",
)

llama_norm_mappings = [
    NormMapping(
        norm="re:.*input_layernorm$",
        linears=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    NormMapping(
        norm="re:.*post_attention_layernorm$",
        linears=["re:.*up_proj$", "re:.*gate_proj$"],
    )
]

class SpinquantRotation(Enum):
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"

class SpinQuantModifier(Modifier):
    rotations: List[SpinquantRotation] = Field(default_factory=lambda: ["R1", "R2"])

    transform_type: Literal["hadamard", "random-hadamard", "random-matrix"] = Field(default="hadamard")
    randomize: bool = Field(default=False)
    learnable: bool = Field(default=False)

    mappings: Optional[SpinQuantMappings] = None
    norm_mappings: Optional[List[NormMapping]] = None
    
    transform_config: Optional[TransformConfig] = None  # optional override for more fine-grained control

    def on_initialize(self, state: State, **kwargs) -> bool:
        # HARDCODE
        self.mappings = llama_spinquant
        self.norm_mappings = llama_norm_mappings

        if self.transform_config is not None:
            if self.mappings is not None:
                raise ValueError()
            
            return True

        config_groups = {}
        for rotation in self.rotations:
            if rotation == SpinquantRotation.R1:
                config_groups["R1"] = self._create_r1_scheme()

            if rotation == SpinquantRotation.R2:
                config_groups["R2"] = self._create_r2_scheme(state.model)

            if rotation == SpinquantRotation.R3:
                config_groups["R3"] = self._create_r3_scheme()

            if rotation == SpinquantRotation.R4:
                config_groups["R4"] = self._create_r4_scheme()

        self.transform_config = TransformConfig(config_groups=config_groups)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        for layer in state.model.model.layers:
            fuse_norm_linears(layer.input_layernorm, (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj))
            fuse_norm_linears(layer.post_attention_layernorm, (layer.mlp.gate_proj, layer.mlp.up_proj))

        # needs to happen after the model has been hooked to execute on the GPU
        # otherwise we're applying weight transforms on CPU
        apply_transform_config(state.model, self.transform_config)




    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        elif event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            pass

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, None)

        return True


    def _create_r1_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            apply=[
                TransformArgs(
                    targets=[
                        self.mappings.embedding,
                        self.mappings.attn_o,
                        *self.mappings.mlp_out,
                    ],
                    location="weight_output",
                ),
                TransformArgs(
                    targets=[
                        self.mappings.attn_q,
                        self.mappings.attn_k,
                        self.mappings.attn_v,
                        *self.mappings.mlp_in,
                        self.mappings.lm_head
                    ],
                    location="weight_input",
                    inverse=True,
                ),
            ]
        )
    
    def _create_r2_scheme(self, model: PreTrainedModel) -> TransformScheme:
        config = model.config

        if hasattr(config, "head_dim"):
            head_dim = config.head_dim
        elif hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            head_dim = config.hidden_size // config.num_attention_heads
        else:
            raise NotImplementedError()
        
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            head_dim=head_dim,
            apply=[
                TransformArgs(targets=[self.mappings.attn_v], location="weight_output"),
                TransformArgs(
                    targets=[self.mappings.attn_o],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )


    def _create_r3_scheme(self) -> TransformScheme:
        raise NotImplementedError()


    def _create_r4_scheme(self) -> TransformScheme:
        raise NotImplementedError()