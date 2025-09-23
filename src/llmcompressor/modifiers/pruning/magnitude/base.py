import warnings
from typing import Any, Dict, List, Union

from pydantic import field_validator

from llmcompressor.core import Event, EventType, ModelParameterizedLayer, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.helpers import (
    PruningCreateSettings,
    PruningSchedulerFactory,
    SchedulerCalculationType,
)
from llmcompressor.modifiers.pruning.utils.pytorch import (
    LayerParamMasking,
    MaskCreatorType,
    PruningMaskCreatorArgs,
    PruningMaskFactory,
)
from llmcompressor.utils.pytorch.module import get_layers_params

__all__ = ["MagnitudePruningModifier"]


class MagnitudePruningModifier(Modifier, LayerParamMasking):
    targets: Union[str, List[str]]
    init_sparsity: float
    final_sparsity: float
    update_scheduler: str = "cubic"
    scheduler_args: Dict[str, Any] = {}
    mask_structure: str = "unstructured"
    leave_enabled: bool = False
    apply_globally: bool = False

    parameterized_layers_: Dict[str, ModelParameterizedLayer] = None
    _save_masks: bool = False
    _use_hooks: bool = False
    scheduler_function_: SchedulerCalculationType = None
    mask_creator_function_: MaskCreatorType = None
    current_sparsity_: float = None

    @field_validator("leave_enabled")
    def validate_leave_enabled(value: bool) -> bool:
        if value:
            warnings.warn(
                "MagnitudePruningModifier.leave_enabled has been deprecated "
                "and will be set to False.",
                DeprecationWarning,
            )
        return False

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.apply_globally:
            raise NotImplementedError("global pruning not implemented yet for PyTorch")

        if "save_masks" in kwargs:
            self._save_masks = kwargs["save_masks"]
        if "use_hooks" in kwargs:
            self._use_hooks = kwargs["use_hooks"]

        if not state.model:
            return False

        self.scheduler_function_ = PruningSchedulerFactory.create_scheduler(
            self.update_scheduler,
            PruningCreateSettings(
                self.start,
                self.end,
                self.update,
                self.init_sparsity,
                self.final_sparsity,
                self.scheduler_args,
            ),
        )
        self.mask_creator_function_ = PruningMaskFactory.create_mask_creator(
            self.mask_structure
        )

        self.parameterized_layers_ = get_layers_params(state.model)

        for layer_param_name, parameterized_layer in self.parameterized_layers_.items():
            self.add_mask(
                layer_param_name,
                parameterized_layer,
                persistent=self._save_masks,
                add_hooks=self._use_hooks,
            )

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        for layer_param_name, _ in self.parameterized_layers_.items():
            self.remove_mask(layer_param_name)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        sparsity = self.scheduler_function_(event, state)
        self.current_sparsity_ = sparsity

        for layer_param_name, parameterized_layer in self.parameterized_layers_.items():
            mask = self.mask_creator_function_(
                PruningMaskCreatorArgs(
                    parameter=parameterized_layer.param,
                    sparsity=sparsity,
                    scores=parameterized_layer.param.data.abs(),
                )
            )
            self.update_mask(layer_param_name, mask)

        self.enable_masks()

    def on_update(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.BATCH_START:
            sparsity = self.scheduler_function_(event, state)
            if sparsity != self.current_sparsity_:
                self.current_sparsity_ = sparsity

                for (
                    layer_param_name,
                    parameterized_layer,
                ) in self.parameterized_layers_.items():
                    mask = self.mask_creator_function_(
                        PruningMaskCreatorArgs(
                            parameter=parameterized_layer.param,
                            sparsity=sparsity,
                            scores=parameterized_layer.param.data.abs(),
                        )
                    )
                    self.update_mask(layer_param_name, mask)
        else:
            self._update_masks(event)

    def on_end(self, state: State, event: Event, **kwargs):
        self.disable_masks()

    def _update_masks(self, event: Event):
        if event.type_ == EventType.OPTIM_PRE_STEP and not self._use_hooks:
            for layer_param_name, _ in self.parameterized_layers_.items():
                self.apply_mask_gradient(layer_param_name)
        elif event.type_ == EventType.OPTIM_POST_STEP and not self._use_hooks:
            for layer_param_name, _ in self.parameterized_layers_.items():
                self.apply_mask_weight(layer_param_name)
