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
    Distills an unquantized teacher into a quantized model by training only the
    gains (``weight``) of its normalization layers. Every other tensor, including
    packed quantized weights and their qparams, is left untouched, so the result
    stays a drop-in replacement for the input checkpoint. The method follows
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
        if not self._match_gains(state.model):
            raise ValueError(
                f"NormDistillationModifier found no normalization gains matching "
                f"targets={self.targets}"
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

    def on_finalize(self, state: State, **kwargs) -> bool:
        # Train after every other modifier has finished compressing the model, and
        # outside the calibration contexts, which disable gradients and the lm_head
        student = state.model
        # Match again: oneshot swaps offset norms for standard norms during
        # calibration and restores the original modules before finalization
        modules = self._match_gains(student)
        batches = [
            {
                key: batch[key].cpu()
                for key in ("input_ids", "attention_mask")
                if key in batch
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
            self._train(student, teacher, modules, train, validation)
        finally:
            for parameter, flag in requires_grad.items():
                parameter.grad = None
                parameter.requires_grad_(flag)
            student.train(student_training)
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

    def _train(self, student, teacher, modules, train, validation):
        gains = [module.weight for module in modules]
        # FP32 master gains: Adam's small updates fall below the FP16/BF16 ULP
        masters = [g.detach().float().clone().requires_grad_(True) for g in gains]
        optimizer = torch.optim.Adam(masters, lr=self.lr)
        scaler = torch.amp.GradScaler(
            get_execution_device(student).type,
            enabled=any(g.dtype == torch.float16 for g in gains),
        )
        for gain in gains:
            gain.requires_grad_(True)

        best_loss = self._evaluate(student, teacher, validation)
        best = [m.detach().clone() for m in masters]
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
                if updates == 0 and any(g.grad is None for g in gains):
                    raise ValueError(
                        "Norm distillation: no gradient reached the normalization "
                        "gains; the student must not be offloaded"
                    )
                for gain, master in zip(gains, masters):
                    master.grad = gain.grad.float()
                    gain.grad = None
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # GradScaler skips the update on overflow and lowers its scale
                skipped += scaler.is_enabled() and scaler.get_scale() < scale
                updates += 1
                self._copy_gains(modules, masters)

            loss = self._evaluate(student, teacher, validation)
            if loss < best_loss:
                best_loss = loss
                best = [m.detach().clone() for m in masters]
            logger.info(
                "Norm distillation epoch {} validation KL {:.6e}, best {:.6e}",
                epoch,
                loss,
                best_loss,
            )

        self._copy_gains(modules, best)
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
    inputs = {key: value.to(device) for key, value in batch.items()}
    logits = model(**inputs, use_cache=False).logits
    # FP32 keeps the softmax and KL reductions accurate for FP16/BF16 logits
    return logits.float().log_softmax(-1)
