from typing import Optional

from compressed_tensors.transform import TransformConfig, apply_transform_config
from pydantic import ValidationError, model_validator

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.transform.presets import TRANSFORM_PRESETS


class TransformModifier(Modifier):
    preset_config: Optional[str] = None
    config: Optional[TransformConfig] = None

    # model validator to validate both preset and config groups are not provided
    @model_validator(mode="after")
    def validate_model_after(model: "TransformModifier") -> "TransformModifier":
        if model.preset_config is None and model.config is None:
            raise ValidationError("Either a config or a preset_config must be provided")

        if model.preset_config is not None:
            if model.preset_config not in TRANSFORM_PRESETS:
                raise ValidationError(
                    f"Invalid preset_config '{model.preset_config}' "
                    f"must be in {TRANSFORM_PRESETS.keys()}"
                )
            model.config = TRANSFORM_PRESETS[model.preset_config]

    def on_initialize(self, state: State, **kwargs) -> bool:
        apply_transform_config(state.model, self.config)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

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
