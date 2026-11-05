import gc
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from compressed_tensors.offload import (
    get_cache_init_kwargs,
    get_execution_device,
    offload_module,
    update_offload_parameter,
)
from compressed_tensors.offload.cache import DiskCache, OffloadCache
from compressed_tensors.offload.module import remove_module_offload
from compressed_tensors.quantization import QuantizationStatus, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import match_named_modules
from loguru import logger
from pydantic import ConfigDict, Field, field_serializer

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["NormDistillationModifier"]


class NormDistillationModifier(Modifier):
    """
    Distills an unquantized teacher into a quantized model by training the gains
    (``weight``) of its normalization layers, its weight quantization scales, or
    both. Quantized weight values stay untouched, so the result stays a drop-in
    replacement for the input checkpoint. The method follows
    Norm Tweaking (https://arxiv.org/abs/2309.02784), with the whole model's
    next-token distribution as the target.

    The main use is improving a checkpoint that is already compressed: load it
    with ``AutoModelForCausalLM.from_pretrained`` (without decompressing it) and
    pass it to ``oneshot`` along with the unquantized model as ``teacher``. The
    modifier can also follow a weight quantization modifier in the same recipe.

    Training runs once all other modifiers have finalized. The loss is the
    token-averaged KL divergence from the teacher's next-token distribution to the
    student's, over the calibration batches. The modifier holds out a fraction of
    the batches and restores the parameters with the lowest validation loss, which
    may be the original ones.

    The student and teacher run full forward and backward passes, so both must
    fit on the execution device. An offloaded student is onloaded for training and
    offloaded again afterwards.

    Sample usage::

        from transformers import AutoModelForCausalLM

        from llmcompressor import oneshot
        from llmcompressor.modifiers.qad import NormDistillationModifier

        model = AutoModelForCausalLM.from_pretrained(
            "RedHatAI/Qwen3-0.6B-quantized.w4a16", dtype="auto", device_map="cuda"
        )
        oneshot(
            model=model,
            dataset=dataset,
            recipe=NormDistillationModifier(teacher="Qwen/Qwen3-0.6B"),
            output_dir="Qwen3-0.6B-quantized.w4a16-norm-distilled",
        )

    :param teacher: unquantized model, or a path or Hugging Face stub to load it
        from. A loaded teacher is used as is; a path is loaded in the student's
        dtype onto the student's execution device
    :param targets: normalization modules whose ``weight`` is trained. Modules
        without a 1-D floating point ``weight`` are skipped. Defaults to every
        module whose name ends in ``norm``
    :param ignore: modules to exclude from ``targets``
    :param tune_norms: whether to train normalization gains. Defaults to True
    :param tune_scales: whether to train the ``weight_scale`` of every
        weight-quantized module. Each scale is trained as a multiplier of its
        original value, with the quantized weight values held fixed. Global scales
        are not trained, since fused projections must share them. Defaults to False
    :param num_epochs: training epochs over the calibration batches. Defaults to 1
    :param lr: Adam learning rate for the gains and the scale multipliers.
        Defaults to 1e-3
    :param gradient_accumulation_steps: calibration batches per optimizer update.
        Defaults to 1
    :param validation_fraction: fraction of calibration batches held out from
        training to select the best epoch; at least one batch is held out.
        Defaults to 0.1
    :param seed: seed for the validation split and per-epoch shuffling.
        Defaults to 42
    """

    # Training reads the calibration batches from state.data.calib and needs no
    # calibration forward passes, so on its own the modifier runs with the
    # data-free pipeline. Already-compressed models decompress on their first
    # forward pass, which the sequential pipeline's subgraphs would skip.
    requires_calibration_data: bool = False

    # Allow arbitrary types because teacher may be a torch.nn.Module
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    teacher: str | torch.nn.Module
    targets: str | list[str] = Field(default_factory=lambda: ["re:.*norm$"])
    ignore: list[str] = Field(default_factory=list)
    tune_norms: bool = True
    tune_scales: bool = False
    num_epochs: int = Field(default=1, ge=1)
    lr: float = Field(default=1.0e-3, gt=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)
    seed: int = 42

    @field_serializer("teacher")
    def _serialize_teacher(self, teacher):
        # Recipes record a loaded teacher by the path it was loaded from
        if isinstance(teacher, torch.nn.Module):
            return getattr(getattr(teacher, "config", None), "_name_or_path", None)
        return teacher

    def on_initialize(self, state: State, **kwargs) -> bool:
        if state.data.calib is None:
            raise ValueError("NormDistillationModifier requires a calibration dataset")
        if not self.tune_norms and not self.tune_scales:
            raise ValueError(
                "NormDistillationModifier must tune norms, scales, or both"
            )
        if self.tune_norms and not self._match_gains(state.model):
            raise ValueError(
                f"NormDistillationModifier found no normalization gains matching "
                f"targets={self.targets}"
            )
        if self.tune_scales and not self._match_scales(state.model):
            raise ValueError(
                "NormDistillationModifier found no weight quantization scales"
            )
        return True

    def _match_gains(self, model):
        return [
            module
            for _, module in match_named_modules(model, self.targets, self.ignore)
            if isinstance(getattr(module, "weight", None), torch.nn.Parameter)
            and module.weight.dim() == 1
            and module.weight.is_floating_point()
        ]

    @staticmethod
    def _match_scales(model):
        # Compressed modules hold `weight_packed` (or a quantized `weight`) until
        # their first forward pass decompresses them
        return [
            module
            for module in model.modules()
            if getattr(module, "quantization_scheme", None) is not None
            and module.quantization_scheme.weights is not None
            and not module.quantization_scheme.weights.dynamic
            and isinstance(getattr(module, "weight_scale", None), torch.Tensor)
            and module.weight_scale.is_floating_point()
            and (
                getattr(module, "weight", None) is not None
                or hasattr(module, "weight_packed")
            )
        ]

    def on_finalize(self, state: State, **kwargs) -> bool:
        # Train after every other modifier has finished compressing the model, and
        # outside the calibration contexts, which disable gradients and the lm_head
        student = state.model
        batches = [
            {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            for batch in state.data.calib
        ]
        train, validation = self._split_batches(batches)
        teacher = self._load_teacher(student)
        student_training, teacher_training = student.training, teacher.training
        student.eval()
        teacher.eval()
        try:
            # The first forward pass decompresses an already-compressed student,
            # which may change how it is offloaded
            with torch.no_grad():
                _log_probs(student, validation[0])
            with _onloaded(student):
                # Evaluate onloaded: an offloaded module's fake-quantized forward
                # pass can onload its weight again without quantization
                initial_loss = self._evaluate(student, teacher, validation)
                # Match again: oneshot swaps offset norms for standard norms during
                # calibration and restores the original modules before finalization
                modules = self._match_gains(student) if self.tune_norms else []
                scaled = (
                    [_ScaledWeight(m) for m in self._match_scales(student)]
                    if self.tune_scales
                    else []
                )
                if not modules and not scaled:
                    raise ValueError(
                        "NormDistillationModifier found no trainable targets"
                    )
                logger.info(
                    "Norm distillation: {} gains, {} weight scales, {} train and {} "
                    "validation batches",
                    len(modules),
                    len(scaled),
                    len(train),
                    len(validation),
                )
                requires_grad = {p: p.requires_grad for p in student.parameters()}
                student.requires_grad_(False)
                try:
                    self._train(
                        student,
                        teacher,
                        modules,
                        scaled,
                        train,
                        validation,
                        initial_loss,
                    )
                finally:
                    for parameter, flag in requires_grad.items():
                        parameter.grad = None
                        parameter.requires_grad_(flag)
        finally:
            student.train(student_training)
            if isinstance(self.teacher, str):
                del teacher
                gc.collect()
                if torch.accelerator.is_available():
                    torch.accelerator.empty_cache()
            else:
                teacher.train(teacher_training)
        return True

    def _load_teacher(self, student):
        if isinstance(self.teacher, torch.nn.Module):
            return self.teacher
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            self.teacher,
            dtype=student.dtype,
            device_map=get_execution_device(student),
        )

    def _split_batches(self, batches):
        if len(batches) < 2:
            raise ValueError(
                "Norm distillation validation requires at least two calibration batches"
            )
        validation_count = min(
            len(batches) - 1, max(1, math.ceil(len(batches) * self.validation_fraction))
        )
        order = torch.randperm(
            len(batches), generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        return (
            [batches[i] for i in order[validation_count:]],
            [batches[i] for i in order[:validation_count]],
        )

    def _train(
        self, student, teacher, modules, scaled, train, validation, initial_loss
    ):
        gains = [module.weight for module in modules]
        for scale in scaled:
            scale.snap()
        weights = [scale.module.weight for scale in scaled]
        trainable = gains + weights
        for parameter in trainable:
            parameter.requires_grad_(True)

        # FP32 master gains: Adam's small updates fall below the FP16/BF16 ULP.
        # Scales train as FP32 multipliers of their original values.
        masters = [g.detach().float().clone().requires_grad_(True) for g in gains]
        ratios = [scale.ratio.clone().requires_grad_(True) for scale in scaled]
        optimizer = torch.optim.Adam(masters + ratios, lr=self.lr)
        scaler = torch.amp.GradScaler(
            get_execution_device(student).type,
            enabled=any(p.dtype == torch.float16 for p in trainable),
        )

        best_loss = initial_loss
        best = self._snapshot(masters, ratios)
        logger.info("Norm distillation initial validation KL {:.6e}", best_loss)
        generator = torch.Generator().manual_seed(self.seed)
        size = self.gradient_accumulation_steps
        updates = skipped = 0
        for epoch in range(1, self.num_epochs + 1):
            order = torch.randperm(len(train), generator=generator).tolist()
            for start in range(0, len(order), size):
                group = [train[i] for i in order[start : start + size]]
                with torch.enable_grad():
                    for batch in group:
                        loss = self._batch_loss(student, teacher, batch) / len(group)
                        scaler.scale(loss).backward()
                if updates == 0 and all(p.grad is None for p in trainable):
                    raise ValueError(
                        "Norm distillation: no gradient reached any trainable target"
                    )
                for gain, master in zip(gains, masters):
                    master.grad = None if gain.grad is None else gain.grad.float()
                for scale, ratio in zip(scaled, ratios):
                    ratio.grad = scale.ratio_grad()
                for parameter in trainable:
                    parameter.grad = None
                loss_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # GradScaler skips the update on overflow and lowers its scale
                skipped += scaler.is_enabled() and scaler.get_scale() < loss_scale
                updates += 1
                self._apply(modules, masters, scaled, ratios)

            loss = self._evaluate(student, teacher, validation)
            if loss < best_loss:
                best_loss = loss
                best = self._snapshot(masters, ratios)
            logger.info(
                "Norm distillation epoch {} validation KL {:.6e}, best {:.6e}",
                epoch,
                loss,
                best_loss,
            )

        self._apply(modules, best[0], scaled, best[1])
        logger.info(
            "Norm distillation: {} updates ({} skipped on overflow), best validation "
            "KL {:.6e}",
            updates,
            skipped,
            best_loss,
        )

    @staticmethod
    def _snapshot(masters, ratios):
        return (
            [m.detach().clone() for m in masters],
            [r.detach().clone() for r in ratios],
        )

    @staticmethod
    @torch.no_grad()
    def _apply(modules, gains, scaled, ratios):
        for module, gain in zip(modules, gains):
            update_offload_parameter(module, "weight", gain.to(module.weight.dtype))
        for scale, ratio in zip(scaled, ratios):
            ratio.clamp_(min=_MIN_RATIO)
            scale.apply(ratio)

    @torch.no_grad()
    def _evaluate(self, student, teacher, batches):
        return sum(
            self._batch_loss(student, teacher, batch).item() for batch in batches
        ) / len(batches)

    @staticmethod
    def _batch_loss(student, teacher, batch):
        with torch.no_grad():
            target = _log_probs(teacher, batch)
        prediction = _log_probs(student, batch)
        token_kl = F.kl_div(
            prediction, target.to(prediction.device), log_target=True, reduction="none"
        ).sum(-1)
        mask = batch.get("attention_mask")
        if mask is None:
            return token_kl.mean()
        mask = mask.to(token_kl.device, token_kl.dtype)
        return (token_kl * mask).sum() / mask.sum().clamp_min(1)


# Keeps trained scales positive
_MIN_RATIO = 1e-3


class _ScaledWeight:
    """
    Trains a module's ``weight_scale`` as a multiplier of its original value while
    its quantized weight values stay fixed.

    The weight the forward pass uses is ``dequantized * ratio`` (expanded to the
    weight's shape), so the gradient of each multiplier is the sum of
    ``weight.grad * dequantized`` over the weights that share its scale. That sum
    is taken over the scale's calibration layout from ``flatten_for_calibration``,
    which covers channel, group, tensor-group and block strategies.
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.args = module.quantization_scheme.weights
        self.scale = module.weight_scale.detach().float().clone()
        self.ratio = torch.ones_like(self.scale)
        self.dequantized = None

    @torch.no_grad()
    def snap(self):
        """
        Record the dequantized weight and use it as the module's weight, so the
        weight the forward pass sees and its gradient are those of the quantized
        values. Called once the module has been decompressed.
        """
        module = self.module
        weight = module.weight.detach()
        if weight.dim() != 2:
            raise ValueError(
                f"Scale tuning supports 2-D weights, got shape {tuple(weight.shape)}"
            )
        status = getattr(module, "quantization_status", None)
        if (
            status is not None
            and status < QuantizationStatus.COMPRESSED
            and getattr(module, "quantization_enabled", True)
        ):
            # Fake-quantized forward: snap the weight to its quantized values,
            # which compression maps back to the same quantized values
            weight = forward_quantize(module, weight, "weight", self.args)
            update_offload_parameter(module, "weight", weight)
        self.dequantized = weight.clone()

    def ratio_grad(self):
        grad = self.module.weight.grad
        if grad is None:
            return None
        product = grad.float() * self.dequantized.to(grad.device).float()
        flat = flatten_for_calibration(product, "weight", self.args)
        return flat.sum(-1).reshape(self.scale.shape).to(self.scale.device)

    @torch.no_grad()
    def apply(self, ratio: torch.Tensor):
        module = self.module
        update_offload_parameter(
            module, "weight_scale", (self.scale * ratio).to(module.weight_scale.dtype)
        )
        # The stored scale may be rounded (FP8, FP16, BF16): scale the weight by
        # the stored multiplier so the forward pass matches the saved checkpoint
        stored = module.weight_scale.detach().float().to(self.scale.device)
        effective = torch.where(self.scale != 0, stored / self.scale, ratio)
        weight = self.dequantized.float() * self.expand(effective).to(
            self.dequantized.device
        )
        update_offload_parameter(module, "weight", weight.to(self.dequantized.dtype))

    def expand(self, value: torch.Tensor) -> torch.Tensor:
        """Expand a tensor shaped like the scale to the weight's shape"""
        rows, cols = self.dequantized.shape
        strategy = self.args.strategy
        if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
            value = value.repeat_interleave(self.args.group_size, dim=-1)
        elif strategy == QuantizationStrategy.BLOCK:
            height, width = self.args.block_structure
            value = value.repeat_interleave(height, dim=0)
            value = value.repeat_interleave(width, dim=1)
        if value.dim() == 2:
            value = value[:rows, :cols]
        return value.expand(rows, cols)


@contextmanager
def _onloaded(model: torch.nn.Module):
    """
    Onload an offloaded model for training, then offload it again with the trained
    values. Gradients do not reach parameters that are onloaded per forward pass.
    """
    offloaded = {}
    for module in model.modules():
        if not isinstance(module._parameters, OffloadCache):
            continue
        if isinstance(module._parameters, DiskCache):
            raise ValueError("Norm distillation does not support disk offloading")
        offloaded[module] = (
            get_cache_init_kwargs(module),
            dict(module._parameters.offloaded_values),
            dict(module._buffers.offloaded_values),
        )
    for module in offloaded:
        remove_module_offload(module, onload_tensors=True)

    try:
        yield
    finally:
        with torch.no_grad():
            for module, (kwargs, parameters, buffers) in offloaded.items():
                # Write the trained values into the original offloaded tensors,
                # which keeps tensors shared between modules (tied embeddings) shared
                for store, originals in (
                    (module._parameters, parameters),
                    (module._buffers, buffers),
                ):
                    for name, current in list(store.items()):
                        original = originals.get(name)
                        if (
                            original is None
                            or current is None
                            or current is original
                            or original.shape != current.shape
                            or original.dtype != current.dtype
                        ):
                            # Unchanged, or replaced during training (decompression)
                            continue
                        original.copy_(current)
                        store[name] = original
                offload_module(module, **kwargs)


def _log_probs(model, batch):
    device = get_execution_device(model)
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    logits = model(**inputs, use_cache=False).logits
    # FP32 keeps the softmax and KL reductions accurate for FP16/BF16 logits
    return logits.float().log_softmax(-1)
