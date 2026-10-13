import gc
import math

import torch
import torch.nn.functional as F
from compressed_tensors.offload import get_execution_device, update_offload_parameter
from compressed_tensors.utils import match_named_modules
from loguru import logger
from pydantic import ConfigDict, Field, field_serializer

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier

__all__ = ["NormDistillationModifier"]


class NormDistillationModifier(Modifier):
    """
    Distills an unquantized teacher into a quantized model by training the gains
    (``weight``) of its normalization layers, its quantization scales, or both.
    Packed quantized weights stay untouched, so the result stays a drop-in
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
    the batches and restores the gains with the lowest validation loss, which may
    be the original gains.

    The student and teacher run full forward and backward passes, so both must
    fit on the execution device without offloading.

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
    :param tune_scales: whether to train weight quantization scales and global
        scales. Defaults to False
    :param num_epochs: training epochs over the calibration batches. Defaults to 1
    :param lr: Adam learning rate. Defaults to 1e-3
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
        return [
            (module, name)
            for module in model.modules()
            for name in ("weight_scale", "weight_global_scale")
            if isinstance(getattr(module, name, None), torch.Tensor)
            and (
                getattr(module, "weight", None) is not None
                or hasattr(module, "weight_packed")
            )
            and getattr(module, "quantization_scheme", None) is not None
            and module.quantization_scheme.weights is not None
            and not getattr(module, name).is_meta
            and getattr(module, name).is_floating_point()
        ]

    def on_finalize(self, state: State, **kwargs) -> bool:
        # Train after every other modifier has finished compressing the model, and
        # outside the calibration contexts, which disable gradients and the lm_head
        student = state.model
        # Match again: oneshot swaps offset norms for standard norms during
        # calibration and restores the original modules before finalization
        modules = self._match_gains(student) if self.tune_norms else []
        scales = self._match_scales(student) if self.tune_scales else []
        if not modules and not scales:
            raise ValueError("NormDistillationModifier found no trainable targets")
        batches = [
            {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            for batch in state.data.calib
        ]
        teacher = self._load_teacher(student)
        train, validation = self._split_batches(batches)
        logger.info(
            "Norm distillation: {} gains, {} train and {} validation batches",
            len(modules),
            len(train),
            len(validation),
        )
        student_training, teacher_training = student.training, teacher.training
        student.eval()
        teacher.eval()
        requires_grad = {p: p.requires_grad for p in student.parameters()}
        student.requires_grad_(False)
        try:
            self._train(student, teacher, modules, scales, train, validation)
        finally:
            for parameter, flag in requires_grad.items():
                parameter.grad = None
                parameter.requires_grad_(flag)
            student.train(student_training)
            if isinstance(self.teacher, str):
                del teacher
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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

    def _train(self, student, teacher, modules, scales, train, validation):
        gains = [module.weight for module in modules]
        # FP32 master gains: Adam's small updates fall below the FP16/BF16 ULP
        masters = [g.detach().float().clone().requires_grad_(True) for g in gains]
        scale_values = [getattr(module, name) for module, name in scales]
        scale_masters = [
            s.detach().float().clone().requires_grad_(True) for s in scale_values
        ]
        optimizer = torch.optim.Adam(masters + scale_masters, lr=self.lr)
        device_type = get_execution_device(student).type
        scaler_enabled = (
            any(g.dtype == torch.float16 for g in gains) and device_type == "cuda"
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        for gain in gains:
            gain.requires_grad_(True)
        weight_bases = {}
        scale_bases = {}

        best_loss = self._evaluate(student, teacher, validation)
        best = [m.detach().clone() for m in masters]
        best_scales = [m.detach().clone() for m in scale_masters]
        if scales:
            for module, name in scales:
                weight_bases[module] = module.weight.detach().clone()
            scale_bases = [s.detach().clone() for s in scale_values]
            for module in weight_bases:
                module.weight.requires_grad_(True)
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
                if updates == 0 and all(
                    parameter.grad is None
                    for parameter in gains + [m.weight for m in weight_bases]
                ):
                    raise ValueError(
                        "Norm distillation: no gradient reached any trainable target; "
                        "the student must not be offloaded"
                    )
                for gain, master in zip(gains, masters):
                    master.grad = gain.grad.float() if gain.grad is not None else None
                    gain.grad = None
                for (module, name), scale, master in zip(
                    scales, scale_values, scale_masters
                ):
                    grad = module.weight.grad
                    if grad is None:
                        master.grad = None
                        continue
                    expanded = self._expand_scale(module, name, scale, module.weight)
                    sign = -1 if name == "weight_global_scale" else 1
                    scale_grad = (
                        grad.float()
                        * module.weight.detach().float()
                        / expanded.float()
                        * sign
                    )
                    master.grad = self._reduce_scale_grad(scale_grad, scale)
                for module in weight_bases:
                    module.weight.grad = None
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # GradScaler skips the update on overflow and lowers its scale
                skipped += scaler.is_enabled() and scaler.get_scale() < scale
                updates += 1
                self._copy_gains(modules, masters)
                for master in scale_masters:
                    master.data.clamp_(min=torch.finfo(master.dtype).eps)
                self._copy_scales(scales, scale_masters)
                self._refresh_scaled_weights(weight_bases, scales, scale_bases)

            loss = self._evaluate(student, teacher, validation)
            if loss < best_loss:
                best_loss = loss
                best = [m.detach().clone() for m in masters]
                best_scales = [m.detach().clone() for m in scale_masters]
            logger.info(
                "Norm distillation epoch {} validation KL {:.6e}, best {:.6e}",
                epoch,
                loss,
                best_loss,
            )

        self._copy_gains(modules, best)
        self._copy_scales(scales, best_scales)
        self._refresh_scaled_weights(weight_bases, scales, scale_bases)
        logger.info(
            "Norm distillation: {} updates ({} skipped on overflow), best validation "
            "KL {:.6e}",
            updates,
            skipped,
            best_loss,
        )

    @staticmethod
    @torch.no_grad()
    def _copy_gains(modules, values):
        for module, value in zip(modules, values):
            update_offload_parameter(module, "weight", value.to(module.weight.dtype))

    @staticmethod
    @torch.no_grad()
    def _copy_scales(scales, values):
        for (module, name), value in zip(scales, values):
            update_offload_parameter(
                module, name, value.to(getattr(module, name).dtype)
            )

    @staticmethod
    def _expand_scale(module, name, scale, weight):
        scale = scale.to(weight.device)
        if name == "weight_global_scale":
            return scale.expand_as(weight)
        strategy = getattr(module.quantization_scheme.weights, "strategy", None)
        strategy_name = getattr(strategy, "value", str(strategy)).lower()
        if scale.ndim == 1 and weight.ndim == 2 and scale.shape[0] == weight.shape[0]:
            scale = scale[:, None]
        if strategy_name.endswith("group"):
            scale = scale.repeat_interleave(
                module.quantization_scheme.weights.group_size, dim=-1
            )
        elif strategy_name.endswith("block"):
            rows, cols = module.quantization_scheme.weights.block_structure
            scale = scale.repeat_interleave(rows, dim=0).repeat_interleave(cols, dim=1)
        return scale.expand_as(weight)

    @staticmethod
    def _reduce_scale_grad(grad, scale):
        return grad.reshape(*scale.shape, -1).sum(-1).reshape_as(scale)

    @staticmethod
    @torch.no_grad()
    def _refresh_scaled_weights(weight_bases, scales, scale_bases):
        for module, base in weight_bases.items():
            current = torch.ones_like(base)
            initial = torch.ones_like(base)
            for (scale_module, name), base_scale in zip(scales, scale_bases):
                if scale_module is module:
                    current_scale = NormDistillationModifier._expand_scale(
                        module, name, getattr(module, name), base
                    )
                    initial_scale = NormDistillationModifier._expand_scale(
                        module, name, base_scale, base
                    )
                    if name == "weight_global_scale":
                        current = current / current_scale
                        initial = initial / initial_scale
                    else:
                        current = current * current_scale
                        initial = initial * initial_scale
            update_offload_parameter(
                module, "weight", base * current / initial.clamp_min(1e-12)
            )

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


def _log_probs(model, batch):
    device = get_execution_device(model)
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    logits = model(**inputs, use_cache=False).logits
    # FP32 keeps the softmax and KL reductions accurate for FP16/BF16 logits
    return logits.float().log_softmax(-1)
