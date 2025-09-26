from enum import Enum
from typing import Iterable, List, Literal, Optional

import torch
from compressed_tensors import match_modules_set, match_named_modules
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import TorchDtype
from pydantic import Field, ValidationInfo, field_validator
from transformers import PreTrainedModel

from llmcompressor.core import Event, EventType, State
from llmcompressor.modeling import center_embeddings, fuse_norm_linears
from llmcompressor.modifiers import Modifier

from .mappings import SpinQuantMapping, infer_mapping_from_model
from .norm_mappings import NormMapping, infer_norm_mapping_from_model


class SpinquantRotation(str, Enum):
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"


class SpinQuantModifier(Modifier, use_enum_values=True):
    """
    Implements the transforms according to "SpinQuant: LLM quantization
    with learned rotations" (https://arxiv.org/abs/2405.16406)

    Transforms (rotations) are extra layers added to a model which reduce the accuracy
    loss induced by quantization. This is achived through "rotating" weights and
    activations into a space with a smaller dynamic range of values, thus decreasing
    the range of scales required for quantization.

    The SpinQuant authors describe four different rotations which can be applied to a
    model. R1 and R2 are "offline" rotations, meaning that they can be fused into
    existing weights and therefore do not induce runtime cost. R3 and R4 are "online"
    rotations, meaning that they require additional computation at runtime.

    Lifecycle:
        - on_initialize
            - infer SpinQuantMappings & NormMappings
            - as needed, create transform schemes for R1, R2, R3, & R4
        - on_start
            - normalize embeddings
            - fuse norm layers into subsequent Linear layers
            - apply TransformConfig
                - fuse transforms into weights for mergeable transforms
                - add hooks for online transforms
        - on sequential epoch end
        - on_end
        - on_finalize

    :param rotations: A list containing the names of rotations to apply to the model.
        Possible rotations include R1, R2, R3, and R4
    :param transform_type: The type of transform to apply to the model.
        `"hadamard"` has the least performance cost but only supports sizes which are
        powers of power of two.
        `"random-matrix"` has more performance cost, but supports a much larger set of
            sizes.
        `"random-matrix"` has the greatest performance cost, but supports any size
    :param randomize: if True, create distinct transforms for each application
    :param learnable: if True, attach gradients to transform weights for training
    :param precision: Precision at which all transforms should be applied. This applies
        to both weight fusing and online rotations
    :param transform_block_size: Block size to use for rotation matrices. The model's
        hidden_size and head_dim must be evenly divisible by transform_block_size.
        Layers will be transformed by a block-diagonal matrix where each block is a
        matrix of this size.
        If None is provided, model's hidden_size will be used for R1, R3, and R4
        and model's head_dim will be used for R2
    :param mappings: Specifies layers within a model to target for transforms.
        A mapping will be inferred if None is provided
    :param norm_mappings: Specifies layers within a model to target for norm fusing.
        A mapping will be inferred if None is provided
    :param transform_config: Optional transform config for overriding provided arguments
    """

    rotations: List[SpinquantRotation] = Field(default_factory=lambda: ["R1", "R2"])
    transform_type: Literal["hadamard", "random-hadamard", "random-matrix"] = Field(
        default="hadamard"
    )
    randomize: bool = Field(default=False)
    learnable: bool = Field(default=False)
    precision: TorchDtype = Field(default=torch.float64)
    transform_block_size: Optional[int] = Field(default=None)

    # norm mappings separate from spinquant mappings to allow users to
    # override spinquant mappings with transform_config without overriding norms
    mappings: Optional[SpinQuantMapping] = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    norm_mappings: Optional[List[NormMapping]] = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    # optional override for more fine-grained control
    # also included in recipe serialization
    transform_config: Optional[TransformConfig] = Field(default=None, repr=False)

    @field_validator("randomize", "learnable", mode="before")
    def validate_not_implemented(cls, value, info: ValidationInfo):
        if value:
            raise NotImplementedError(f"{info.field_name} is not supported as of now")
        return value

    @field_validator("rotations", mode="before")
    def validate_rotations(cls, value):
        if isinstance(value, Iterable):
            return tuple(v.upper() for v in value)
        return value

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.transform_config is not None:
            return True

        self.mappings = infer_mapping_from_model(state.model)
        self.norm_mappings = infer_norm_mapping_from_model(state.model)

        config_groups = {}
        if SpinquantRotation.R1 in self.rotations:
            config_groups["R1"] = self._create_r1_scheme()

        if SpinquantRotation.R2 in self.rotations:
            config_groups["R2"] = self._create_r2_scheme(state.model)

        if SpinquantRotation.R3 in self.rotations:
            config_groups["R3"] = self._create_r3_scheme()

        if SpinquantRotation.R4 in self.rotations:
            config_groups["R4"] = self._create_r4_scheme()

        self.transform_config = TransformConfig(config_groups=config_groups)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # needs to happen after the model has been hooked to execute on the GPU
        # otherwise we're applying weight transforms on CPU
        self._center_embeddings(state.model)
        self._fuse_norms(state.model)
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

    def _center_embeddings(self, model: PreTrainedModel):
        for _, embedding in match_named_modules(
            model, [self.mappings.embedding], warn_on_fail=True
        ):
            center_embeddings(embedding)

    def _fuse_norms(self, model: PreTrainedModel):
        for mapping in self.norm_mappings:
            for norm, *linears in match_modules_set(
                model, (mapping.norm, *mapping.linears)
            ):
                fuse_norm_linears(norm, linears)

    def _create_r1_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
            head_dim=self.transform_block_size,
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
                        self.mappings.lm_head,
                    ],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _create_r2_scheme(self, model: PreTrainedModel) -> TransformScheme:
        config = model.config

        if hasattr(config, "head_dim"):
            head_dim = config.head_dim
        elif hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            head_dim = config.hidden_size // config.num_attention_heads
        else:
            raise NotImplementedError()

        if self.transform_block_size:
            if head_dim % self.transform_block_size != 0:
                raise ValueError(
                    f"transform_block_size {self.transform_block_size} must be set "
                    f"such that model's head_dim {head_dim} is evenly divisible by it"
                )
            head_dim = self.transform_block_size

        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
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
        raise NotImplementedError(
            "SpinQuant R3 rotations will be added in a future release"
        )

    def _create_r4_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
            head_dim=self.transform_block_size,
            apply=[
                TransformArgs(
                    targets=[*self.mappings.mlp_out],
                    location="input",
                ),
                TransformArgs(
                    targets=[*self.mappings.mlp_out],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )
