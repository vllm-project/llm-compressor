import gc
import math
from dataclasses import dataclass
from typing import Any

import torch
from compressed_tensors.offload import get_execution_device
from compressed_tensors.quantization import enable_quantization
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.quantization.utils import is_module_quantized
from compressed_tensors.utils import getattr_chain, update_offload_parameter
from loguru import logger
from pydantic import Field, PrivateAttr

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.utils.pytorch import infer_sequential_targets

__all__ = ["LayerwiseQADModifier"]


@dataclass
class _LayerBatch:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    loss_mask: torch.Tensor | None


def _map_tensors(value: Any, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, transform) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, transform) for key, item in value.items()}
    return value


def _extract_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _extract_hidden(output[0])
    if isinstance(output, dict):
        for key in ("last_hidden_state", "hidden_states"):
            if key in output:
                return _extract_hidden(output[key])
    for key in ("last_hidden_state", "hidden_states"):
        if hasattr(output, key):
            return _extract_hidden(getattr(output, key))
    raise TypeError(
        f"Unable to extract a hidden state from {type(output).__name__} output"
    )


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "Student and teacher hidden-state shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )

    per_token_loss = (prediction.float() - target.float()).pow(2).mean(dim=-1)
    if loss_mask is None:
        return per_token_loss.mean()

    mask = loss_mask.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
    while mask.ndim < per_token_loss.ndim:
        mask = mask.unsqueeze(-1)
    try:
        mask = torch.broadcast_to(mask, per_token_loss.shape)
    except RuntimeError as error:
        raise ValueError(
            f"Loss mask shape {tuple(loss_mask.shape)} cannot broadcast to "
            f"hidden-state loss shape {tuple(per_token_loss.shape)}"
        ) from error

    denominator = mask.sum()
    if denominator.item() == 0:
        raise ValueError("Loss mask does not contain any valid tokens")
    return (per_token_loss * mask).sum() / denominator


class LayerwiseQADModifier(Modifier):
    """
    Optimizes one fake-quantized decoder block at a time against the matching
    full-precision teacher block using hidden-state mean squared error.

    This modifier is intended to follow a quantization modifier such as
    ``GPTQModifier`` in the same recipe. The preceding modifier must initialize
    quantization parameters for the current block before layerwise QAD runs.

    ``num_epochs`` controls how many times the cached calibration batches are
    traversed for each block. ``gradient_accumulation_steps`` controls how many
    cached batches contribute to one optimizer update. Calibration dataloader
    ``batch_size`` is therefore the QAD microbatch size.
    """

    requires_calibration_data: bool = True

    num_epochs: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2.0e-6, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float | None = Field(default=1.0, gt=0)
    seed: int = 42
    target_offload_device: str = "cpu"
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)
    early_stopping_patience: int = Field(default=3, ge=1)
    validation_relative_min_delta: float = Field(default=1.0e-3, ge=0)

    _capture_enabled: bool = PrivateAttr(default=False)
    _layer_inputs: dict[torch.nn.Module, list[_LayerBatch]] = PrivateAttr(
        default_factory=dict
    )
    _module_names: dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _student_to_teacher: dict[torch.nn.Module, torch.nn.Module] = PrivateAttr(
        default_factory=dict
    )
    _sequential_targets: list[str] = PrivateAttr(default_factory=list)
    _state: State | None = PrivateAttr(default=None)
    _optimizer_steps: dict[str, int] = PrivateAttr(default_factory=dict)
    _best_validation_losses: dict[str, float] = PrivateAttr(default_factory=dict)
    _epochs_completed: dict[str, int] = PrivateAttr(default_factory=dict)
    _validation_histories: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    @property
    def optimizer_steps(self) -> dict[str, int]:
        return dict(self._optimizer_steps)

    @property
    def best_validation_losses(self) -> dict[str, float]:
        return dict(self._best_validation_losses)

    @property
    def epochs_completed(self) -> dict[str, int]:
        return dict(self._epochs_completed)

    @property
    def validation_histories(self) -> dict[str, list[float]]:
        return {
            name: list(history) for name, history in self._validation_histories.items()
        }

    def on_initialize(self, state: State, **kwargs) -> bool:
        if state.teacher_model is None:
            raise ValueError(
                "LayerwiseQADModifier requires a full-precision teacher model. "
                "Pass `distill_teacher` to `oneshot` or `teacher_model` to "
                "`CompressionSession.initialize`."
            )
        if state.teacher_model is state.model:
            raise ValueError("Teacher and student must be different model instances")

        self._state = state
        self._sequential_targets = infer_sequential_targets(
            state.model,
            sequential_targets=kwargs.get("sequential_targets"),
        )
        teacher_modules = dict(state.teacher_model.named_modules())

        for name, module in state.model.named_modules():
            if module.__class__.__name__ not in self._sequential_targets:
                continue
            if name not in teacher_modules:
                raise ValueError(f"Teacher model is missing decoder block `{name}`")
            teacher_module = teacher_modules[name]
            if teacher_module.__class__ is not module.__class__:
                raise ValueError(
                    f"Teacher and student block types differ for `{name}`: "
                    f"{teacher_module.__class__.__name__} != "
                    f"{module.__class__.__name__}"
                )
            self._module_names[module] = name
            self._student_to_teacher[module] = teacher_module
            self._layer_inputs[module] = []

        if not self._student_to_teacher:
            raise ValueError(
                "LayerwiseQADModifier could not find any sequential decoder blocks"
            )

        state.teacher_model.eval()
        state.teacher_model.requires_grad_(False)
        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        self._capture_enabled = True
        for module in self._student_to_teacher:
            self.register_hook(
                module,
                self._capture_input,
                "forward_pre",
                with_kwargs=True,
            )

    def _capture_input(self, module, args, kwargs):
        if not self._capture_enabled:
            return

        loss_mask = None
        if self._state is not None and self._state.loss_masks is not None:
            batch_index = self._state.current_batch_idx
            if batch_index < 0 or batch_index >= len(self._state.loss_masks):
                raise RuntimeError(
                    f"Invalid calibration batch index {batch_index} for loss masks"
                )
            loss_mask = self._state.loss_masks[batch_index]

        def detach(tensor):
            return tensor.detach().to(self.target_offload_device)

        self._layer_inputs[module].append(
            _LayerBatch(
                args=_map_tensors(args, detach),
                kwargs=_map_tensors(kwargs, detach),
                loss_mask=None if loss_mask is None else detach(loss_mask),
            )
        )

    def on_sequential_epoch_end(
        self,
        state: State,
        event: Event,
        modules: list[torch.nn.Module],
        **kwargs,
    ):
        decoder_blocks = [
            module for module in modules if module in self._student_to_teacher
        ]
        if not decoder_blocks:
            return
        if len(decoder_blocks) != 1:
            raise ValueError(
                "LayerwiseQADModifier requires one decoder block per sequential "
                f"subgraph, found {len(decoder_blocks)}"
            )

        student_block = decoder_blocks[0]
        batches = self._layer_inputs[student_block]
        if not batches:
            name = self._module_names.get(
                student_block,
                student_block.__class__.__name__,
            )
            raise ValueError(f"No calibration inputs were captured for block `{name}`")

        self._capture_enabled = False
        try:
            with HooksMixin.disable_hooks():
                self._optimize_block(
                    student_block,
                    self._student_to_teacher[student_block],
                    batches,
                )
        finally:
            self._layer_inputs[student_block].clear()
            self._capture_enabled = True

    def _optimize_block(
        self,
        student_block: torch.nn.Module,
        teacher_block: torch.nn.Module,
        batches: list[_LayerBatch],
    ):
        name = self._module_names.get(
            student_block,
            student_block.__class__.__name__,
        )
        self._log_device_memory(name, "before cache release")
        self._release_device_cache()
        self._log_device_memory(name, "after cache release")
        trainable = self._get_trainable_weights(student_block)
        if not trainable:
            raise ValueError(
                f"Block `{name}` has no quantized floating-point weights to optimize. "
                "Ensure LayerwiseQADModifier follows GPTQModifier in the recipe."
            )

        device = get_execution_device(student_block)
        teacher_parameters = list(teacher_block.parameters())
        if not teacher_parameters:
            raise ValueError(f"Teacher block `{name}` has no parameters")
        teacher_device = teacher_parameters[0].device
        if any(parameter.device != teacher_device for parameter in teacher_parameters):
            raise ValueError(
                f"Teacher block `{name}` must reside on a single device before QAD"
            )
        teacher_dtype = teacher_parameters[0].dtype

        teacher_block.to(device)
        teacher_block.eval()
        student_block.eval()
        student_block.apply(enable_quantization)

        for parameter in student_block.parameters():
            parameter.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)

        try:
            train_indices, validation_indices = self._split_batch_indices(len(batches))
            initial_train_loss = self._evaluate_indices(
                student_block,
                teacher_block,
                batches,
                train_indices,
                device,
                teacher_dtype,
            )
            initial_validation_loss = self._evaluate_indices(
                student_block,
                teacher_block,
                batches,
                validation_indices,
                device,
                teacher_dtype,
            )
            self._release_device_cache()
            self._log_device_memory(name, "after initial evaluation")
            optimizer = torch.optim.AdamW(
                trainable,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            optimizer_steps, epochs_completed, best_validation_loss = (
                self._train_with_validation(
                    student_block,
                    teacher_block,
                    batches,
                    optimizer,
                    trainable,
                    train_indices,
                    validation_indices,
                    device,
                    teacher_dtype,
                )
            )
            final_train_loss = self._evaluate_indices(
                student_block,
                teacher_block,
                batches,
                train_indices,
                device,
                teacher_dtype,
            )
            final_validation_loss = self._evaluate_indices(
                student_block,
                teacher_block,
                batches,
                validation_indices,
                device,
                teacher_dtype,
            )
            self._materialize_quantized_weights(student_block)
            self._optimizer_steps[name] = optimizer_steps
            self._epochs_completed[name] = epochs_completed
            self._best_validation_losses[name] = best_validation_loss
            logger.info(
                "Layerwise QAD optimized {} for {} steps over {} epochs: "
                "train MSE {:.6e} -> {:.6e}, validation MSE {:.6e} -> {:.6e}",
                name,
                optimizer_steps,
                epochs_completed,
                initial_train_loss,
                final_train_loss,
                initial_validation_loss,
                final_validation_loss,
            )
        finally:
            for parameter in student_block.parameters():
                parameter.requires_grad_(False)
            teacher_block.to(teacher_device)
            self._release_device_cache()

    def _get_trainable_weights(
        self, block: torch.nn.Module
    ) -> list[torch.nn.Parameter]:
        trainable = []
        for module in block.modules():
            weights = getattr_chain(module, "quantization_scheme.weights", None)
            if not is_module_quantized(module) or weights is None:
                continue
            parameter = getattr(module, "weight", None)
            scale = getattr(module, "weight_scale", None)
            if not isinstance(parameter, torch.nn.Parameter):
                continue
            if not parameter.is_floating_point():
                continue
            if scale is None or scale.device.type == "meta":
                raise ValueError(
                    "Quantization parameters are not initialized. Place "
                    "LayerwiseQADModifier after GPTQModifier in the recipe."
                )
            trainable.append(parameter)
        return trainable

    def _train(
        self,
        student_block: torch.nn.Module,
        teacher_block: torch.nn.Module,
        batches: list[_LayerBatch],
        optimizer: torch.optim.Optimizer,
        trainable: list[torch.nn.Parameter],
        device: torch.device,
        teacher_dtype: torch.dtype,
    ) -> int:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        optimizer_steps = 0

        with torch.enable_grad():
            for _ in range(self.num_epochs):
                indices = torch.randperm(len(batches), generator=generator).tolist()
                for start in range(0, len(indices), self.gradient_accumulation_steps):
                    accumulation_group = indices[
                        start : start + self.gradient_accumulation_steps
                    ]
                    optimizer.zero_grad(set_to_none=True)
                    for batch_index in accumulation_group:
                        loss = self._batch_loss(
                            student_block,
                            teacher_block,
                            batches[batch_index],
                            device,
                            teacher_dtype,
                        )
                        (loss / len(accumulation_group)).backward()

                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            trainable,
                            self.max_grad_norm,
                        )
                    optimizer.step()
                    optimizer_steps += 1

        return optimizer_steps

    def _train_with_validation(
        self,
        student_block: torch.nn.Module,
        teacher_block: torch.nn.Module,
        batches: list[_LayerBatch],
        optimizer: torch.optim.Optimizer,
        trainable: list[torch.nn.Parameter],
        train_indices: list[int],
        validation_indices: list[int],
        device: torch.device,
        teacher_dtype: torch.dtype,
    ) -> tuple[int, int, float]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        optimizer_steps = 0
        epochs_completed = 0
        epochs_without_improvement = 0
        best_validation_loss = self._evaluate_indices(
            student_block,
            teacher_block,
            batches,
            validation_indices,
            device,
            teacher_dtype,
        )
        patience_reference_loss = best_validation_loss
        best_weights = self._snapshot_weights(trainable)
        name = self._module_names.get(
            student_block,
            student_block.__class__.__name__,
        )
        validation_history = [best_validation_loss]

        for epoch in range(1, self.num_epochs + 1):
            shuffled = torch.randperm(
                len(train_indices),
                generator=generator,
            ).tolist()
            epoch_indices = [train_indices[index] for index in shuffled]
            optimizer_steps += self._train_epoch(
                student_block,
                teacher_block,
                batches,
                optimizer,
                trainable,
                epoch_indices,
                device,
                teacher_dtype,
            )
            epochs_completed += 1
            validation_loss = self._evaluate_indices(
                student_block,
                teacher_block,
                batches,
                validation_indices,
                device,
                teacher_dtype,
            )
            validation_history.append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_weights = self._snapshot_weights(trainable)

            relative_improvement = (patience_reference_loss - validation_loss) / max(
                abs(patience_reference_loss), torch.finfo(torch.float32).tiny
            )
            if relative_improvement >= self.validation_relative_min_delta:
                patience_reference_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            logger.info(
                "Layerwise QAD {} epoch {} validation MSE {:.6e}, "
                "best {:.6e}, patience {}/{}",
                name,
                epoch,
                validation_loss,
                best_validation_loss,
                epochs_without_improvement,
                self.early_stopping_patience,
            )
            if epochs_without_improvement >= self.early_stopping_patience:
                break

        self._restore_weights(trainable, best_weights)
        self._validation_histories[name] = validation_history
        return optimizer_steps, epochs_completed, best_validation_loss

    def _train_epoch(
        self,
        student_block: torch.nn.Module,
        teacher_block: torch.nn.Module,
        batches: list[_LayerBatch],
        optimizer: torch.optim.Optimizer,
        trainable: list[torch.nn.Parameter],
        indices: list[int],
        device: torch.device,
        teacher_dtype: torch.dtype,
    ) -> int:
        optimizer_steps = 0
        with torch.enable_grad():
            for start in range(0, len(indices), self.gradient_accumulation_steps):
                accumulation_group = indices[
                    start : start + self.gradient_accumulation_steps
                ]
                optimizer.zero_grad(set_to_none=True)
                for batch_index in accumulation_group:
                    loss = self._batch_loss(
                        student_block,
                        teacher_block,
                        batches[batch_index],
                        device,
                        teacher_dtype,
                    )
                    (loss / len(accumulation_group)).backward()

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable,
                        self.max_grad_norm,
                    )
                optimizer.step()
                optimizer_steps += 1
        return optimizer_steps

    def _split_batch_indices(self, batch_count: int) -> tuple[list[int], list[int]]:
        if batch_count < 2:
            raise ValueError(
                "Layerwise QAD validation requires at least two calibration batches"
            )
        validation_count = min(
            batch_count - 1,
            max(1, math.ceil(batch_count * self.validation_fraction)),
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        shuffled = torch.randperm(batch_count, generator=generator).tolist()
        validation_indices = shuffled[:validation_count]
        train_indices = shuffled[validation_count:]
        return train_indices, validation_indices

    def _snapshot_weights(
        self,
        trainable: list[torch.nn.Parameter],
    ) -> list[torch.Tensor]:
        return [
            parameter.detach().to(self.target_offload_device).clone()
            for parameter in trainable
        ]

    @staticmethod
    @torch.no_grad()
    def _restore_weights(
        trainable: list[torch.nn.Parameter],
        weights: list[torch.Tensor],
    ):
        for parameter, weight in zip(trainable, weights):
            parameter.copy_(weight.to(device=parameter.device, dtype=parameter.dtype))

    @torch.no_grad()
    def _evaluate(
        self,
        student_block: torch.nn.Module,
        teacher_block: torch.nn.Module,
        batches: list[_LayerBatch],
        device: torch.device,
        teacher_dtype: torch.dtype,
    ) -> float:
        losses = [
            self._batch_loss(
                student_block,
                teacher_block,
                batch,
                device,
                teacher_dtype,
            ).item()
            for batch in batches
        ]
        return sum(losses) / len(losses)

    @torch.no_grad()
    def _evaluate_indices(
        self,
        student_block: torch.nn.Module,
        teacher_block: torch.nn.Module,
        batches: list[_LayerBatch],
        indices: list[int],
        device: torch.device,
        teacher_dtype: torch.dtype,
    ) -> float:
        losses = [
            self._batch_loss(
                student_block,
                teacher_block,
                batches[index],
                device,
                teacher_dtype,
            ).item()
            for index in indices
        ]
        return sum(losses) / len(losses)

    def _batch_loss(
        self,
        student_block: torch.nn.Module,
        teacher_block: torch.nn.Module,
        batch: _LayerBatch,
        device: torch.device,
        teacher_dtype: torch.dtype,
    ) -> torch.Tensor:
        def move(tensor):
            return tensor.to(device=device)

        args = _map_tensors(batch.args, move)
        kwargs = _map_tensors(batch.kwargs, move)
        with torch.no_grad():
            teacher_args = _map_tensors(
                args,
                lambda tensor: tensor.to(
                    dtype=(
                        teacher_dtype if tensor.is_floating_point() else tensor.dtype
                    )
                ),
            )
            teacher_kwargs = _map_tensors(
                kwargs,
                lambda tensor: tensor.to(
                    dtype=(
                        teacher_dtype if tensor.is_floating_point() else tensor.dtype
                    )
                ),
            )
            target = _extract_hidden(teacher_block(*teacher_args, **teacher_kwargs))
        prediction = _extract_hidden(student_block(*args, **kwargs))
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        return _masked_mse(prediction, target, batch.loss_mask)

    @torch.no_grad()
    def _materialize_quantized_weights(self, block: torch.nn.Module):
        for module in block.modules():
            weights = getattr_chain(module, "quantization_scheme.weights", None)
            if not is_module_quantized(module) or weights is None:
                continue
            quantized_weight = forward_quantize(
                module,
                module.weight,
                "weight",
                weights,
            )
            update_offload_parameter(module, "weight", quantized_weight)

    @staticmethod
    def _release_device_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _log_device_memory(name: str, stage: str):
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(
            "Layerwise QAD memory for {} {}: {:.2f} GiB allocated, {:.2f} GiB reserved",
            name,
            stage,
            allocated,
            reserved,
        )

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self._capture_enabled = False
        self.remove_hooks()
        self._layer_inputs.clear()

    def on_finalize(self, state: State, **kwargs) -> bool:
        self._state = None
        return True
