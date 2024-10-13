from typing import Any, Dict, List, Tuple, Union

from torch.nn import Module

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.distillation.utils.pytorch import (
    KDFactory,
    KDModelWrapper,
    KDModuleWrapper,
)
from llmcompressor.utils.fsdp.context import summon_full_params_context
from llmcompressor.utils.fsdp.helpers import maybe_get_wrapped, set_wrapped_model
from llmcompressor.utils.pytorch.module import get_layers, set_layer

__all__ = ["OutputDistillationModifier"]


class OutputDistillationModifier(Modifier):
    targets: Union[str, List[Union[str, Tuple[str, str]]]]
    projection: str = None
    projection_args: Dict[str, Any] = None
    transforms: Union[str, List[str]] = "identity"
    transforms_args: Union[Dict[str, Any], List[Dict[str, Any]]] = None
    comparison: str = "kl_divergence"
    comparison_args: Dict[str, Any] = None
    orig_scale: float = 1.0
    distill_scale: float = 1.0
    offload_layer_output: bool = False

    wrappers_: Dict[str, Any] = None
    wrapped_kd_model_: Any = None
    fsdp_active_: bool = False

    def on_initialize(self, state: State, **kwargs) -> bool:
        if state.model is None or state.teacher_model is None:
            return False

        self.wrappers_ = {}
        if kwargs.get("fsdp_active"):
            self.fsdp_active_ = True

        if not hasattr(state.model.config, "hidden_size"):
            raise ValueError(
                "Model config must specify hidden_size in order to use "
                "OutputDistillationModifier"
            )

        # needed to initialize intermediate output buffers for student and teacher
        hidden_size = (
            kwargs.get("metadata").get("per_device_train_batch_size", 1),
            kwargs.get("metadata").get("max_seq_length", 512),
            state.model.config.hidden_size,
        )

        for target in (
            self.targets if isinstance(self.targets, list) else [self.targets]
        ):
            if isinstance(target, tuple):
                model_target, teacher_target = target
            else:
                model_target, teacher_target = target, target

            model_layers = get_layers(model_target, state.model)
            teacher_layers = get_layers(teacher_target, state.teacher_model)

            if len(model_layers) < 1:
                raise ValueError(f"no model layers found for target {target}")

            if len(model_layers) != len(teacher_layers):
                raise ValueError(
                    f"model and teacher model layers for target {target} do not match"
                )

            for (key, student_layer), teacher_layer in zip(
                model_layers.items(), teacher_layers.values()
            ):
                student_wrapper = self._create_layer_wrapper(
                    student_layer, hidden_size, state
                )
                teacher_wrapper = self._create_layer_wrapper(
                    teacher_layer, hidden_size, state
                )
                self.wrappers_[key] = (student_wrapper, teacher_wrapper)

        with summon_full_params_context(state.teacher_model, offload_to_cpu=True):
            for key, (student_wrapper, teacher_wrapper) in self.wrappers_.items():
                set_layer(key, student_wrapper, state.model)
                set_layer(key, teacher_wrapper, state.teacher_model)

        self.wrapped_kd_model_ = self._create_model_wrapper(
            student_model=maybe_get_wrapped(state.model),
            teacher_model=state.teacher_model,
            state=state,
        )

        set_wrapped_model(state, self.wrapped_kd_model_)

        # for square-head distillation we want to scale the loss by the number of
        # layers if the user doesn't alter the default scale. This is done so the
        # distillation loss is roughly equally weighted to the cross entropy loss
        num_layers = len(self.wrappers_)
        if self.comparison == "square_head" and self.distill_scale == 1.0:
            self.distill_scale = float(num_layers)
        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        set_wrapped_model(state, self.wrapped_kd_model_.student_model)

        with summon_full_params_context(state.teacher_model, offload_to_cpu=True):
            for key, (student_wrapper, teacher_wrapper) in self.wrappers_.items():
                set_layer(key, student_wrapper.layer, state.model)
                set_layer(key, teacher_wrapper.layer, state.teacher_model)
                del student_wrapper
                del teacher_wrapper

        del self.wrapped_kd_model_
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        for student_wrapper, teacher_wrapper in self.wrappers_.values():
            student_wrapper.kd_enabled = True
            teacher_wrapper.kd_enabled = True
        self.wrapped_kd_model_.kd_enabled = True

    def on_update(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.LOSS_CALCULATED and event.should_update(
            self.start, self.end, self.update
        ):
            distill_loss = self.wrapped_kd_model_.kd_last_comparison
            model_loss = self.orig_scale * kwargs["loss"]
            distill_loss = self.distill_scale * distill_loss.to(model_loss.device)
            state.loss = model_loss + distill_loss

    def on_end(self, state: State, event: Event, **kwargs):
        for student_wrapper, teacher_wrapper in self.wrappers_.values():
            student_wrapper.kd_enabled = False
            teacher_wrapper.kd_enabled = False
        self.wrapped_kd_model_.kd_enabled = False

    def _create_model_wrapper(
        self, student_model: Module, teacher_model: Module, state: State
    ) -> KDModelWrapper:
        comparison = KDFactory.create_comparison(
            self.comparison,
            student_model,
            teacher_model,
            state,
            **(self.comparison_args or {}),
        )

        return KDModelWrapper(
            student_model=student_model,
            teacher_model=teacher_model,
            wrappers=self.wrappers_,
            comparison=comparison,
            fsdp_active=self.fsdp_active_,
        )

    def _create_layer_wrapper(
        self, layer: Module, hidden_size: int, state: State
    ) -> KDModuleWrapper:
        transforms = []
        if self.transforms:
            tmp_transforms = (
                self.transforms
                if isinstance(self.transforms, list)
                else [self.transforms]
            )
            tmp_transform_args = [
                args
                for args in (
                    self.transforms_args
                    if isinstance(self.transforms_args, list)
                    else [self.transforms_args if self.transforms_args else {}]
                )
                for _ in range(len(tmp_transforms))
            ]

            for transform, transform_args in zip(tmp_transforms, tmp_transform_args):
                transforms.append(
                    KDFactory.create_transform(
                        transform,
                        layer,
                        state,
                        **transform_args,
                    )
                )

        return KDModuleWrapper(
            layer=layer,
            hidden_size=hidden_size,
            transforms=transforms,
            fsdp_active=self.fsdp_active_,
            offload_output=self.offload_layer_output,
        )
