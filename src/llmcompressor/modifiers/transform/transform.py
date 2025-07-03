from typing import Dict, Optional

from compressed_tensors.transform import TransformScheme, apply_transform_config

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier

from .template.quip import QUIP


class TransformModifier(Modifier):
    preset_config: Optional[str] = None
    config_groups: Optional[Dict[str, TransformScheme]] = None

    # model validator to validate both preset and config groups are not provided

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.preset_config is not None:
            # import config template and customize to model
            pass

        # config = TransformConfig(config_groups=self.config_groups)
        config = QUIP

        apply_transform_config(state.model, config)

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
