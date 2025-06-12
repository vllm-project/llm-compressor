from typing import Dict, Optional

from compressed_tensors.transform import TransformScheme, apply_transform_config

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier

from .template.quip import QUIP


class TransformModifier(Modifier):
    preset_config: Optional[str] = None
    config_groups: Optional[Dict[str, TransformScheme]] = None

    # model validator to validate both preset and config gropus are not provided

    def on_initialize(self, state: State, **kwargs):
        if self.preset_config is not None:
            # import config template and customize to model
            pass

        # config = TransformConfig(config_groups=self.config_groups)
        config = QUIP

        apply_transform_config(state.model, config)
        breakpoint()
