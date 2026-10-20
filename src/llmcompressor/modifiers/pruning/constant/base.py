import torch
from pydantic import Field

from llmcompressor.core import Event, EventType, ModelParameterizedLayer, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.utils.pytorch import (
    LayerParamMasking,
    param_mask_name,
)
from llmcompressor.utils.pytorch.module import build_parameterized_layers

__all__ = ["ConstantPruningModifier"]


class ConstantPruningModifier(Modifier, LayerParamMasking):
    targets: str | list[str]
    parameterized_layers_: dict[str, ModelParameterizedLayer] = Field(
        default_factory=dict
    )
    _epsilon: float = 10e-9
    _save_masks: bool = False
    _use_hooks: bool = False

    def on_initialize(self, state: State, **kwargs) -> bool:
        if "save_masks" in kwargs:
            self._save_masks = kwargs["save_masks"]
        if "use_hooks" in kwargs:
            self._use_hooks = kwargs["use_hooks"]

        if not state.model:
            return False

        return True

    def _initialize_scope(self, state: State, modules: list[torch.nn.Module] | None):
        for layer_param_name, parameterized_layer in build_parameterized_layers(
            state.model, self.targets, modules=modules
        ).items():
            if layer_param_name in self.parameterized_layers_:
                continue
            self.parameterized_layers_[layer_param_name] = parameterized_layer
            self.add_mask(
                layer_param_name,
                parameterized_layer,
                persistent=self._save_masks,
                add_hooks=self._use_hooks,
            )
            self.update_mask(
                layer_param_name, parameterized_layer.param.data.abs() > self._epsilon
            )
        self.enable_masks()

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        self._initialize_scope(state, kwargs.get("modules"))

    def on_sequential_epoch_start(
        self, state: State, event: Event, modules: list[torch.nn.Module], **kwargs
    ):
        self._initialize_scope(state, modules)

    def on_finalize(self, state: State, **kwargs) -> bool:
        for layer_param_name, _ in self.parameterized_layers_.items():
            self.remove_mask(layer_param_name)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self._initialize_scope(state, None)

    @torch.no_grad()
    def on_update(self, state: State, event: Event, **kwargs):
        if self._use_hooks:
            # hooks are used to update, so nothing to do here
            return
        if event.type_ == EventType.OPTIM_POST_STEP:

            def apply_masks(module):
                mask_name = param_mask_name()
                if hasattr(module, mask_name):
                    mask = getattr(module, mask_name)
                    if mask.device != module.weight.device:
                        setattr(module, mask_name, mask.to(module.weight.device))
                    module.weight *= getattr(module, mask_name)

            state.model.apply(apply_masks)

    def on_end(self, state: State, event: Event, **kwargs):
        self.disable_masks()
