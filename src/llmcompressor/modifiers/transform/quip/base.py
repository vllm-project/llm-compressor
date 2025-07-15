from typing import Iterable, List, Literal, Optional, Union

from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from pydantic import Field, ValidationInfo, field_validator

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier

__all__ = ["QuIPModifier"]


class QuIPModifier(Modifier):
    """
    Implements the transforms according to
    [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/pdf/2402.04396)  # noqa: E501
    [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304)  # noqa: E501

    Transforms (rotations) are extra layers added to a model which reduce the accuracy
    loss induced by quantization. This is achived through "rotating" weights and
    activations into a space with a smaller dynamic range of values, thus decreasing
    the range of scales required for quantization.
    
    QuIP and QuIP# apply transforms to every linear layer, two of which are fused into
    the model weights and two of which remain as online rotations computed at runtime.

    :param transform_type: The type of transform to apply to the model.
        `"hadamard"` has the least performance cost but only supports sizes which are
        powers of power of two.
        `"random-matrix"` has more performance cost, but supports a much larger set of
            sizes.
        `"random-matrix"` has the greatest performance cost, but supports any size
    :param randomize: If true, create distinct transforms for each application
    :param learnable: If true, attach gradients to transform weights for training
    :param ignore: Modules to ignore when attaching transforms
    :param transform_config: Optional transform config for overriding provided arguments
    """

    transform_type: Literal["hadamard", "random-hadamard", "random-matrix"] = Field(
        default="hadamard", exclude=True
    )
    randomize: bool = Field(default=False, exclude=True)
    learnable: bool = Field(default=False, exclude=True)
    ignore: Union[str, List[str]] = Field(default="lm_head", exclude=True)

    # optional override for more fine-grained control
    # also included in recipe serialization
    transform_config: Optional[TransformConfig] = Field(default=None, repr=False)

    @field_validator("randomize", "learnable", mode="before")
    def validate_not_implemented(cls, value, info: ValidationInfo):
        raise NotImplementedError(f"{info.field_name} is not supported right now")

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
        return TransformConfig(
            config_groups={
                "v": TransformScheme(
                    type=self.transform_type,
                    apply=[
                        TransformArgs(
                            targets=["Linear"],
                            location="input",  # non-mergable
                            ignore=self.ignore,
                        ),
                        TransformArgs(
                            targets=["Linear"],
                            location="weight_input",
                            inverse=True,
                            ignore=self.ignore,
                        ),
                    ],
                    randomize=self.randomize,
                    requires_grad=self.learnable,
                ),
                "u": TransformScheme(
                    type=self.transform_type,
                    apply=[
                        TransformArgs(
                            targets=["Linear"],
                            location="weight_output",
                            ignore=self.ignore,
                        ),
                        TransformArgs(
                            targets=["Linear"],
                            location="output",  # non-mergable
                            inverse=True,
                            ignore=self.ignore,
                        ),
                    ],
                    randomize=self.randomize,
                    requires_grad=self.learnable,
                ),
            }
        )
