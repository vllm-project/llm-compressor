from typing import List, Literal, Optional, Union

import torch
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import TorchDtype
from pydantic import Field, ValidationInfo, field_validator

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier

__all__ = ["QuIPModifier"]


class QuIPModifier(Modifier):
    """
    Implements the transforms according to
    [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/pdf/2402.04396)
    [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304)

    Transforms (rotations) are extra layers added to a model which reduce the accuracy
    loss induced by quantization. This is achieved through "rotating" weights and
    activations into a space with a smaller dynamic range of values, thus decreasing
    the range of scales required for quantization.

    QuIP and QuIP# apply transforms to every linear layer, two of which are fused into
    the model weights and two of which remain as online rotations computed at runtime.

    Lifecycle:
        - on_initialize
            - as needed, create transform schemes for V (input) and U (output)
        - on_start
            - apply TransformConfig
                - fuse transforms into weights for mergeable transforms
                - add hooks for online transforms
        - on sequential epoch end
        - on_end
        - on_finalize

    :param rotations: which rotation schemes to apply to the model. Including `"v"` will
        rotate the input side of weights, and including `"u"` will rotate the output
        side of weights (note that v does not require u and vice-versa)
    :param transform_type: The type of transform to apply to the model.
        `"hadamard"` has the least performance cost but only supports sizes which are
        powers of power of two.
        `"random-hadamard"` has more performance cost, but supports a much larger set of
            sizes.
        `"random-matrix"` has the greatest performance cost, but supports any size
    :param randomize: If true, create distinct transforms for each application
    :param learnable: If true, attach gradients to transform weights for training
    :param precision: Precision at which all transforms should be applied. This applies
        to both weight fusing and online rotations
    :param transform_block_size: Block size to use for rotation matrices. The model's
        hidden_size must be evenly divisible by transform_block_size.
        Layers will be transformed by a block-diagonal matrix where each block is a
        matrix of this size.
        If None is provided, model's hidden_size will be used
    :param ignore: Modules to ignore when attaching transforms
    :param transform_config: Optional transform config for overriding provided arguments
    """  # noqa: E501

    rotations: List[Literal["v", "u"]] = Field(default_factory=lambda: ["v", "u"])
    transform_type: Literal["hadamard", "random-hadamard", "random-matrix"] = Field(
        default="random-hadamard"
    )
    targets: Union[List[str], str] = Field(default="Linear")
    randomize: bool = Field(default=False)
    learnable: bool = Field(default=False)
    precision: TorchDtype = Field(default=torch.float64)
    transform_block_size: Optional[int] = Field(default=None)
    ignore: Union[str, List[str]] = Field(default="lm_head")

    # optional override for more fine-grained control
    # also included in recipe serialization
    transform_config: Optional[TransformConfig] = Field(default=None, repr=False)

    @field_validator("randomize", "learnable", mode="before")
    def validate_not_implemented(cls, value, info: ValidationInfo):
        if value:
            raise NotImplementedError(f"{info.field_name} is not supported right now")
        return value

    @field_validator("rotations", mode="before")
    def validate_lowercase_list(cls, value):
        if isinstance(value, list):
            value = [v.lower() if isinstance(v, str) else v for v in value]
        return value

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.transform_config is not None:
            return True

        self.transform_config = self._create_config()
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

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

    def _create_config(self) -> TransformConfig:
        config_groups = dict()
        if "v" in self.rotations:
            config_groups["v"] = self._create_v_scheme()
        if "u" in self.rotations:
            config_groups["u"] = self._create_u_scheme()

        return TransformConfig(config_groups=config_groups)

    def _create_v_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            head_dim=self.transform_block_size,
            apply=[
                TransformArgs(
                    targets=self.targets,
                    location="input",  # non-mergable
                    ignore=self.ignore,
                ),
                TransformArgs(
                    targets=self.targets,
                    location="weight_input",
                    inverse=True,
                    ignore=self.ignore,
                ),
            ],
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
        )

    def _create_u_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            head_dim=self.transform_block_size,
            apply=[
                TransformArgs(
                    targets=self.targets,
                    location="weight_output",
                    ignore=self.ignore,
                ),
                TransformArgs(
                    targets=self.targets,
                    location="output",  # non-mergable
                    inverse=True,
                    ignore=self.ignore,
                ),
            ],
            randomize=self.randomize,
            requires_grad=self.learnable,
            precision=self.precision,
        )
