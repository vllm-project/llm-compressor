import math
from functools import partial

import torch
from compressed_tensors.distributed import is_distributed
from compressed_tensors.offload import disable_onloading, get_execution_device
from compressed_tensors.quantization import QuantizationStatus, enable_quantization
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.registry import standardize_lookup_name
from compressed_tensors.utils import (
    getattr_chain,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import Field, PrivateAttr
from torch.utils._pytree import tree_flatten

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import observe, update_qparams
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.utils.pytorch import infer_sequential_targets

__all__ = ["QADModifier"]


def _mse(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Student and teacher output shapes differ")
    error = (prediction.float() - target.float()).square()
    per_token_loss = error.mean(dim=-1) if error.ndim else error
    return per_token_loss.mean()


def _output_loss(prediction, target):
    """Validate output structure and average MSE over floating output leaves."""
    predictions, prediction_spec = tree_flatten(prediction)
    targets, target_spec = tree_flatten(target)
    if prediction_spec != target_spec:
        raise ValueError("Student and teacher output structures differ")
    losses = []
    for pred, ref in zip(predictions, targets):
        if isinstance(ref, torch.Tensor) and ref.is_floating_point() and ref.numel():
            if not isinstance(pred, torch.Tensor):
                raise ValueError("Student and teacher output structures differ")
            losses.append(_mse(pred, ref))
    if not losses:
        raise ValueError("QAD requires a floating output affected by quantized weights")
    return torch.stack(losses).mean()


def _quantized_modules(modules):
    return [
        module
        for module in dict.fromkeys(modules)
        if getattr_chain(module, "quantization_scheme.weights", None) is not None
    ]


class QADModifier(Modifier):
    """Reconstruct each sequential target's outputs after weight quantization.

    Place after a weight quantization modifier in the recipe. Calibration hooks
    cache the current block's unquantized outputs on the student's inputs. At
    ``sequential_epoch_end``, the preceding modifier initializes quantization,
    then QAD optimizes the block's floating weights with fake quantization.
    Weight quantization parameters are re-observed before training, after each
    epoch, and before final weight materialization. No separate teacher is loaded.

    Use the sequential pipeline with one complete target module per subgraph and
    ``propagate_error=True``. QAD shares the quantizer's calibration batches,
    including preprocessing and microbatch size. Its training/validation split,
    epochs and early stopping apply independently to each block. Validation
    batches are withheld from QAD updates, but still calibrate the quantizer.

    RTN and GPTQ are tested. Other preceding methods must preserve the teacher
    during calibration, initialize qparams before QAD's end callback, and use
    floating weights compatible with compressed-tensors fake quantization.
    """

    requires_calibration_data: bool = True
    reobserve_weights: bool = True
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

    _blocks: dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _captured: dict[torch.nn.Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    _block: torch.nn.Module | None = PrivateAttr(default=None)
    _weight_modules: list[torch.nn.Module] = PrivateAttr(default_factory=list)
    _batches: IntermediatesCache = PrivateAttr(default_factory=IntermediatesCache)
    _sequential_prefetch: bool = PrivateAttr(default=False)
    _name: str = PrivateAttr(default="")
    _device: torch.device = PrivateAttr(default_factory=lambda: torch.device("cpu"))
    _optimizer_steps: dict[str, int] = PrivateAttr(default_factory=dict)
    _best_validation_losses: dict[str, float] = PrivateAttr(default_factory=dict)
    _epochs_completed: dict[str, int] = PrivateAttr(default_factory=dict)
    _validation_histories: dict[str, list[float]] = PrivateAttr(default_factory=dict)
    _reobservations: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    @property
    def optimizer_steps(self):
        return dict(self._optimizer_steps)

    @property
    def best_validation_losses(self):
        return dict(self._best_validation_losses)

    @property
    def epochs_completed(self):
        return dict(self._epochs_completed)

    @property
    def validation_histories(self):
        return {
            name: list(values) for name, values in self._validation_histories.items()
        }

    @property
    def reobservations(self):
        return {name: list(stages) for name, stages in self._reobservations.items()}

    def on_initialize(self, state: State, **kwargs) -> bool:
        pipeline = kwargs.get("pipeline")
        if pipeline and standardize_lookup_name(pipeline) != "sequential":
            raise ValueError(
                'QADModifier requires pipeline="sequential" so the preceding '
                "weight quantization modifier and QAD share each block's "
                f"calibration stage; got pipeline={pipeline!r}"
            )
        if not kwargs.get("propagate_error", True):
            raise ValueError(
                "QADModifier requires propagate_error=True so each block receives "
                "the preceding blocks' final quantized outputs"
            )
        modules = _quantized_modules(state.model.modules())
        if not modules:
            raise ValueError(
                "Place QADModifier after a weight quantization modifier, such as "
                "QuantizationModifier (RTN) or GPTQModifier, in the same recipe"
            )
        if any(
            getattr(m, "quantization_status", None) == QuantizationStatus.COMPRESSED
            for m in modules
        ):
            raise ValueError("QAD needs floating weights before weight compression")
        if is_distributed() and torch.distributed.get_world_size() > 1:
            raise ValueError("QAD currently supports single-process calibration")
        targets = infer_sequential_targets(
            state.model, kwargs.get("sequential_targets")
        )
        self._blocks = {
            module: name or type(module).__name__
            for name, module in match_named_modules(state.model, targets)
            if _quantized_modules(module.modules())
        }
        owners = {}
        covered = set()
        with disable_onloading():
            for block in self._blocks:
                for module in _quantized_modules(block.modules()):
                    owner = owners.setdefault(id(module.weight), block)
                    if owner is not block:
                        raise ValueError(
                            "QAD does not support overlapping targets or quantized "
                            "weights shared across blocks"
                        )
                    covered.add(module)
        if covered != set(modules):
            raise ValueError(
                "QAD requires all quantized weights inside sequential target modules; "
                "ignore the output LM head and select complete decoder blocks"
            )
        return True

    def on_calibration_start(self, state, event, **kwargs):
        for block in self._blocks:
            self.register_hook(
                block,
                partial(self._capture_inputs, state=state),
                "forward_pre",
                with_kwargs=True,
            )
            self.register_hook(block, self._capture_output, "forward", with_kwargs=True)

    def _capture_inputs(self, module, args, kwargs, *, state):
        try:
            index = state.current_batch_idx
            if index < 0:
                raise ValueError('QADModifier requires pipeline="sequential"')
            if module not in self._captured:
                self._captured[module] = IntermediatesCache(
                    offload_device=self.target_offload_device
                )
            batches = self._captured[module]
            if self._blocks[module] in self._optimizer_steps or index != len(batches):
                raise ValueError(
                    "QAD requires each target to run once per calibration batch "
                    "in one sequential stage"
                )
            if any(
                getattr(m, "quantization_enabled", False)
                for m in _quantized_modules(module.modules())
            ):
                raise ValueError("QAD teacher capture requires quantization disabled")
            batches.append({"args": args, "kwargs": kwargs}, copy=True)
        except Exception:
            self._clear_cache()
            raise

    def _capture_output(self, module, args, kwargs, output):
        try:
            batches = self._captured[module]
            batches.update(len(batches) - 1, {"target": output}, copy=True)
        except Exception:
            self._clear_cache()
            raise

    def on_sequential_epoch_end(self, state, event, modules, **kwargs):
        try:
            blocks = [m for m in dict.fromkeys(modules) if m in self._blocks]
            if not blocks:
                return
            if len(blocks) != 1:
                raise ValueError(
                    "QAD requires one target module per subgraph; set "
                    "sequential_targets_per_subgraph=1"
                )
            self._block = blocks[0]
            self._name = self._blocks[self._block]
            self._batches = self._captured.pop(self._block, IntermediatesCache())
            self._sequential_prefetch = state.sequential_prefetch
            if self._sequential_prefetch and torch.accelerator.is_available():
                for index in range(len(self._batches)):
                    self._batches.pin_memory(index)
            self._split_batch_indices(len(self._batches))
            logger.info(
                "QAD cached {} local teacher batches for {}",
                len(self._batches),
                self._name,
            )
            with HooksMixin.disable_hooks():
                self._weight_modules = _quantized_modules(self._block.modules())
                self._optimize_block(self._weight_modules)
        except Exception:
            self._clear_cache()
            raise
        finally:
            self._batches = IntermediatesCache()
            self._sequential_prefetch = False
            self._block = None
            self._weight_modules = []

    def _optimize_block(self, modules):
        trainable = []
        for module in modules:
            if (
                not isinstance(module.weight, torch.nn.Parameter)
                or not module.weight.is_floating_point()
            ):
                raise ValueError(
                    "QAD requires floating-point weights for fake quantization"
                )
            for key in ("weight_scale", "weight_global_scale"):
                value = getattr(module, key, None)
                if key == "weight_global_scale" and value is None:
                    continue
                if (
                    value is None
                    or value.is_meta
                    or not torch.isfinite(value.float()).all()
                    or not (value.float() > 0).all()
                ):
                    raise ValueError(
                        f"QAD requires initialized {key}; run the preceding "
                        "quantization modifier before QAD at the block boundary"
                    )
            trainable.append(module.weight)
        trainable = list(dict.fromkeys(trainable))
        self._device = get_execution_device(modules[0])
        # FP32 optimizer state/master weights avoid Adam's epsilon underflow and
        # sub-ULP updates when the model's execution weights are FP16/BF16.
        masters = [
            torch.nn.Parameter(p.detach().float().clone())
            if p.dtype in (torch.float16, torch.bfloat16)
            else p
            for p in trainable
        ]
        optimizer = torch.optim.AdamW(
            masters, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        flags = {p: p.requires_grad for p in self._block.parameters()}
        self._block.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)
        for module in modules:
            enable_quantization(module)
        try:
            train_indices, validation_indices = self._split_batch_indices(
                len(self._batches)
            )
            self._reobserve_weights("before_training")
            initial_train = self._evaluate(train_indices)
            steps, epochs, best_loss = self._train_with_validation(
                optimizer, trainable, masters, train_indices, validation_indices
            )
            self._reobserve_weights("before_materialization")
            self._materialize_quantized_weights(modules)
            final_train = self._evaluate(train_indices)
            final_validation = self._evaluate(validation_indices)
            self._optimizer_steps[self._name] = steps
            self._epochs_completed[self._name] = epochs
            self._best_validation_losses[self._name] = best_loss
            logger.info(
                "QAD {}: {} updates over {} epochs; train MSE {:.6e} -> {:.6e}, "
                "final validation MSE {:.6e}",
                self._name,
                steps,
                epochs,
                initial_train,
                final_train,
                final_validation,
            )
        finally:
            for parameter, requires_grad in flags.items():
                parameter.grad = None
                parameter.requires_grad_(requires_grad)

    def _train_with_validation(
        self, optimizer, trainable, masters, train_indices, validation_indices
    ):
        generator = torch.Generator().manual_seed(self.seed)
        scaler = torch.amp.GradScaler(
            self._device.type,
            enabled=any(p.dtype == torch.float16 for p in trainable),
        )
        best_loss = self._evaluate(validation_indices)
        reference_loss = best_loss
        best_weights = self._snapshot_weights(trainable)
        best_qparams = self._snapshot_qparams()
        history = [best_loss]
        patience = steps = epochs = 0
        for epoch in range(1, self.num_epochs + 1):
            order = torch.randperm(len(train_indices), generator=generator).tolist()
            steps += self._train_epoch(
                optimizer, trainable, masters, [train_indices[i] for i in order], scaler
            )
            epochs = epoch
            self._reobserve_weights(f"epoch_{epoch}")
            loss = self._evaluate(validation_indices)
            history.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_weights = self._snapshot_weights(trainable)
                best_qparams = self._snapshot_qparams()
            improvement = (reference_loss - loss) / max(
                abs(reference_loss), torch.finfo(torch.float32).tiny
            )
            if improvement >= self.validation_relative_min_delta:
                reference_loss, patience = loss, 0
            else:
                patience += 1
            logger.info(
                "QAD {} epoch {} validation MSE {:.6e}, best {:.6e}, patience {}/{}",
                self._name,
                epoch,
                loss,
                best_loss,
                patience,
                self.early_stopping_patience,
            )
            if patience >= self.early_stopping_patience:
                break
        self._restore_weights(trainable, best_weights)
        for module, qparams in zip(self._weight_modules, best_qparams):
            for key, value in qparams.items():
                current = getattr(module, key)
                update_offload_parameter(
                    module, key, value.to(current.device, copy=True)
                )
        self._validation_histories[self._name] = history
        return steps, epochs, best_loss

    def _snapshot_qparams(self):
        return [
            {
                key: value.detach().to(self.target_offload_device, copy=True)
                for key in ("weight_scale", "weight_zero_point", "weight_global_scale")
                if (value := getattr(module, key, None)) is not None
            }
            for module in self._weight_modules
        ]

    @torch.no_grad()
    def _reobserve_weights(self, stage):
        if not self.reobserve_weights:
            return
        if not self._weight_modules or any(
            getattr(module, "weight_observer", None) is None
            for module in self._weight_modules
        ):
            raise ValueError("QAD weight re-observation requires live weight observers")
        # Observe the entire group before updating scales: fused Q/K/V and
        # gate/up projections can share a global-scale observer.
        observe(self._weight_modules, "weight")
        update_qparams(self._weight_modules, "weight")
        for module in self._weight_modules:
            for key in ("weight_scale", "weight_global_scale", "weight_zero_point"):
                value = getattr(module, key, None)
                if value is not None:
                    numeric = value.float()  # Preserve native FP8 scale storage.
                    if not torch.isfinite(numeric).all() or (
                        "scale" in key and not (numeric > 0).all()
                    ):
                        raise ValueError(f"Invalid {key} after QAD re-observation")
        self._reobservations.setdefault(self._name, []).append(stage)
        logger.info("QAD {} re-observed weights: {}", self._name, stage)

    def _train_epoch(self, optimizer, trainable, masters, indices, scaler):
        steps = 0
        with torch.enable_grad():
            for start in range(0, len(indices), self.gradient_accumulation_steps):
                group = indices[start : start + self.gradient_accumulation_steps]
                # Mean reconstruction losses can underflow during FP16 backward,
                # even with FP32 master weights. Keep the scale fixed throughout
                # accumulation, then unscale the FP32 gradients before clipping.
                for _ in range(32):
                    optimizer.zero_grad(set_to_none=True)
                    for parameter in trainable:
                        parameter.grad = None
                    for batch in self._iter_batches(group):
                        loss = self._batch_loss(batch) / len(group)
                        scaler.scale(loss).backward()
                    for parameter, master in zip(trainable, masters):
                        if parameter.grad is not None and master is not parameter:
                            master.grad = parameter.grad.float()
                    scaler.unscale_(optimizer)
                    if all(
                        p.grad is None or torch.isfinite(p.grad).all() for p in masters
                    ):
                        break
                    if not scaler.is_enabled():
                        raise ValueError(f"Nonfinite QAD gradient in {self._name}")
                    # GradScaler skips the overflowing step and lowers its scale.
                    # Retry the same group so every calibration batch contributes.
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    raise ValueError(f"Nonfinite QAD gradient in {self._name}")
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        masters, self.max_grad_norm, error_if_nonfinite=True
                    )
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    for parameter, master in zip(trainable, masters):
                        if parameter is not master:
                            parameter.copy_(master)
                steps += 1
        return steps

    def _split_batch_indices(self, batch_count):
        if batch_count < 2:
            raise ValueError("QAD validation requires at least two calibration batches")
        validation_count = min(
            batch_count - 1, max(1, math.ceil(batch_count * self.validation_fraction))
        )
        order = torch.randperm(
            batch_count, generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        return order[validation_count:], order[:validation_count]

    def _snapshot_weights(self, parameters):
        return [
            p.detach().to(self.target_offload_device, copy=True) for p in parameters
        ]

    @staticmethod
    @torch.no_grad()
    def _restore_weights(parameters, weights):
        for parameter, weight in zip(parameters, weights):
            parameter.copy_(weight)

    @torch.no_grad()
    def _evaluate(self, indices):
        return sum(
            self._batch_loss(batch).item() for batch in self._iter_batches(indices)
        ) / len(indices)

    def _iter_batches(self, indices):
        # Share offloaded entries while retaining split, shuffle, and retry order.
        cache = IntermediatesCache(
            [self._batches.batch_intermediates[i] for i in indices],
            self._batches.offload_device,
        )
        return cache.iter_prefetch() if self._sequential_prefetch else cache.iter()

    def _batch_loss(self, batch):
        prediction = self._block(*batch["args"], **batch["kwargs"])
        loss = _output_loss(prediction, batch["target"])
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite QAD loss in {self._name}")
        return loss

    @staticmethod
    @torch.no_grad()
    def _materialize_quantized_weights(modules):
        for module in modules:
            weight = forward_quantize(
                module, module.weight, "weight", module.quantization_scheme.weights
            )
            update_offload_parameter(module, "weight", weight)

    def _clear_cache(self):
        self.remove_hooks()
        self._captured.clear()
        self._batches = IntermediatesCache()
        self._sequential_prefetch = False
        self._block = None
        self._weight_modules = []

    def on_calibration_end(self, state, event, **kwargs):
        try:
            if len(self._optimizer_steps) != len(self._blocks):
                raise ValueError(
                    "QAD did not optimize every target; use the sequential pipeline "
                    "with one complete target module per subgraph"
                )
        finally:
            self._clear_cache()

    def on_finalize(self, state, **kwargs):
        self._clear_cache()
        return True
