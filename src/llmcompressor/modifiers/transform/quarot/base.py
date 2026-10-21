# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline QuaRot lifecycle for the explicit-expert GLM MLA topology."""

import weakref
from collections import defaultdict
from time import perf_counter

import torch
import torch.distributed as dist
from compressed_tensors.distributed import is_distributed, is_source_process
from compressed_tensors.offload import (
    OffloadCache,
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)
from compressed_tensors.offload.cache import DistributedCPUCache, DistributedDiskCache
from compressed_tensors.utils import TorchDtype
from loguru import logger
from pydantic import Field, PrivateAttr, field_validator

from llmcompressor.core import Event, State
from llmcompressor.modeling import fuse_norm_linears
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.transform.utils.distributed import finish_transform_phase
from llmcompressor.utils import get_high_precision, untie_word_embeddings

from .mappings import RotationPlan, WeightRotation, build_glm_plan
from .rotation import HadamardRotation, make_hadamard_rotation, rotate_axis

__all__ = ["QuaRotModifier"]


class _QuaRotProgress:
    """Low-frequency host wall-time progress, counted after each yielded job.

    No accelerator synchronization is added. Timings include host work and
    blocking transfers, but are not synchronized device-kernel measurements.
    """

    def __init__(self, phase: str, total: int):
        self.phase, self.total = phase, total
        self.completed = 0
        self.interval = max(1, (total + 19) // 20)
        self.started = perf_counter()
        logger.info("[QuaRot] {} 0/{} | elapsed=0.00 s", phase, total)

    def track(self, items):
        for item in items:
            yield item
            # Resumption means the caller finished the previous consumer/op.
            self.completed += 1
            if self.completed % self.interval == 0 or self.completed == self.total:
                logger.info(
                    "[QuaRot] {} {}/{} | elapsed={:.2f} s",
                    self.phase,
                    self.completed,
                    self.total,
                    perf_counter() - self.started,
                )

    def complete(self):
        logger.info(
            "[QuaRot] {} complete: {:.2f} s",
            self.phase,
            perf_counter() - self.started,
        )


class QuaRotModifier(Modifier):
    """Fuse four offline rotations into a GLM MLA decoder before calibration.

    Currently supports explicit, unsharded routed/shared experts and optional
    indexers, with power-of-two Hadamard blocks. MTP, online rotations, fused
    experts are not supported. Distributed execution uses CT shared CPU/disk
    backing, rotated once by the source rank before calibration. No samples
    are required for this modifier itself.

    Initialization validates topology and constructs matrices without changing
    model parameters. CALIBRATION_START unties a shared embedding/head if needed,
    fuses norm gains into every consumer, and applies all rotation pairs. Repeated
    calibration starts in the same lifecycle are harmless. Persisted config
    metadata prevents a new modifier from rotating the transformed model again.

    :param block_size: repeated Hadamard block size. None uses full dimensions
        except the shifted query-latent rotation, whose base block is 32.
    :param seed: local CPU RNG seed, reset independently for each rotation space
    :param precision: matrix construction and fusion arithmetic dtype. float32
        matches the ModelSlim reference; float64 minimizes offline rounding.
    """

    block_size: int | None = Field(default=None, strict=True, gt=0)
    seed: int = Field(default=1234, strict=True, ge=0, le=2**63 - 1)
    precision: TorchDtype = Field(default=get_high_precision())

    _plan: RotationPlan | None = PrivateAttr(default=None)
    _matrices: dict[str, HadamardRotation] = PrivateAttr(default_factory=dict)
    _model_ref: weakref.ReferenceType | None = PrivateAttr(default=None)
    _applied: bool = PrivateAttr(default=False)

    @field_validator("block_size")
    @classmethod
    def validate_block_size(cls, value):
        if value is not None and value & (value - 1):
            raise ValueError("block_size must be a power of two")
        return value

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, value):
        if value not in (torch.float32, torch.float64):
            raise ValueError("QuaRot precision must be float32 or float64")
        return value

    def _validate_model(self, model: torch.nn.Module) -> RotationPlan:
        if getattr(model.config, "quarot_config", None) is not None:
            raise ValueError(
                "Model already has QuaRot state; use an untransformed model. "
                "A failed transform cannot be retried in place."
            )
        # Inspect cached shapes/identities without loading the whole model from disk.
        with OffloadCache.disable_onloading():
            plan = build_glm_plan(model)
            targets = {op.target for op in (plan.embedding, *plan.rotations)}
            targets.update(fusion.norm for fusion in plan.fusions)
            aliases = defaultdict(list)
            for name, parameter in model.named_parameters(remove_duplicate=False):
                key = (
                    (str(parameter.device), parameter.untyped_storage().data_ptr())
                    if not parameter.is_meta
                    and parameter.layout == torch.strided
                    and parameter.numel()
                    else id(parameter)
                )
                aliases[key].append(name)
            tied_pair = {"model.embed_tokens.weight", "lm_head.weight"}
            for names in aliases.values():
                if len(names) > 1 and any(
                    name.rsplit(".", 1)[0] in targets for name in names
                ):
                    if set(names) != tied_pair:
                        raise ValueError(
                            f"Unsupported shared QuaRot parameters: {names}"
                        )
                    embedding = model.get_submodule("model.embed_tokens")
                    head = model.get_submodule("lm_head")
                    if embedding.weight is not head.weight:
                        raise ValueError(
                            "Shared storage views must be untied before QuaRot"
                        )
                    if any(
                        isinstance(module._parameters, OffloadCache)
                        and str(module._parameters.offload_device) == "disk"
                        for module in (embedding, head)
                    ):
                        raise ValueError(
                            "Untie embeddings before disk offloading for QuaRot"
                        )
                    if not (
                        callable(getattr(model, "get_input_embeddings", None))
                        and callable(getattr(model, "get_output_embeddings", None))
                        and model.get_input_embeddings()
                        is model.get_submodule("model.embed_tokens")
                        and model.get_output_embeddings()
                        is model.get_submodule("lm_head")
                    ):
                        raise ValueError(
                            "Tied embeddings require matching embedding accessors"
                        )
            for name in targets:
                module = model.get_submodule(name)
                if getattr(module, "quantization_status", None) not in (
                    None,
                    "initialized",
                ):
                    raise ValueError("QuaRot must run before quantization calibration")
                # Quantization initialization also registers integer scale metadata.
                # Only the floating weight/bias participate in this transform.
                for parameter_name in ("weight", "bias"):
                    parameter = getattr(module, parameter_name, None)
                    if parameter is None:
                        continue
                    if parameter.dtype not in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                        torch.float64,
                    ):
                        raise ValueError(
                            f"{name}: expected unquantized floating parameters"
                        )
                    if parameter.layout != torch.strided:
                        raise ValueError(f"{name}: sparse parameters are unsupported")
                    if parameter.is_meta and not isinstance(
                        module._parameters, OffloadCache
                    ):
                        raise ValueError(f"{name}: unmaterialized meta parameters")
                if str(get_execution_device(module)) in ("meta", "disk"):
                    raise ValueError(f"{name}: an execution device is required")
        return plan

    def on_initialize(self, state: State, **kwargs) -> bool:
        error, signature = None, None
        try:
            plan = self._validate_model(state.model)
            if is_distributed():
                targets = {op.target for op in (plan.embedding, *plan.rotations)}
                targets.update(fusion.norm for fusion in plan.fusions)
                targets.update(name for f in plan.fusions for name in f.consumers)
                layout = []
                with OffloadCache.disable_onloading():
                    for name in sorted(targets):
                        module = state.model.get_submodule(name)
                        cache = module._parameters
                        if not isinstance(
                            cache, (DistributedCPUCache, DistributedDiskCache)
                        ):
                            raise ValueError(
                                "Distributed QuaRot requires CT shared CPU/disk "
                                f"backing; use load_context + auto_offload: {name}"
                            )
                        layout.append(
                            (
                                name,
                                type(cache).__name__,
                                [
                                    (key, tuple(value.shape), str(value.dtype))
                                    for key in ("weight", "bias")
                                    if (value := getattr(module, key, None)) is not None
                                ],
                            )
                        )
                signature = (
                    plan,
                    self.seed,
                    self.block_size,
                    str(self.precision),
                    layout,
                )
        except Exception as caught:
            error = caught
        if is_distributed():
            peers = [None] * dist.get_world_size()
            dist.all_gather_object(peers, (str(error) if error else None, signature))
            if any(message for message, _ in peers) or any(
                other != signature for _, other in peers
            ):
                raise ValueError(
                    "Invalid or inconsistent QuaRot initialization: "
                    f"{[message for message, _ in peers]}"
                ) from error
        elif error is not None:
            raise error
        matrices = {
            space.name: make_hadamard_rotation(
                space.size,
                block_size=self.block_size,
                shifted=space.shifted,
                seed=self.seed,
                dtype=self.precision,
            )
            for space in plan.spaces
        }
        self._plan, self._matrices = plan, matrices
        self._model_ref = weakref.ref(state.model)
        return True

    @torch.no_grad()
    def _rotate(self, model: torch.nn.Module, operation: WeightRotation):
        module = model.get_submodule(operation.target)
        with align_module_device(module):
            options = dict(
                axis=operation.axis,
                stride=operation.stride,
                offset=operation.offset,
                precision=self.precision,
            )
            matrix = self._matrices[operation.space]
            weight = rotate_axis(module.weight, matrix, **options)
            update_offload_parameter(module, "weight", weight)
            if operation.axis == 0 and getattr(module, "bias", None) is not None:
                bias = rotate_axis(module.bias, matrix, **options)
                update_offload_parameter(module, "bias", bias)

    @torch.no_grad()
    def on_calibration_start(self, state: State, event: Event, **kwargs):
        model = state.model
        if self._model_ref is None or self._model_ref() is not model:
            raise ValueError("QuaRot lifecycle model changed after initialization")
        if self._applied:
            return
        started = perf_counter()
        report_progress = not is_distributed() or is_source_process()
        if report_progress:
            logger.info("[QuaRot] preparation / validation started (host wall time)")
        error = None
        try:
            if self._validate_model(model) != self._plan:
                raise ValueError("QuaRot topology changed after initialization")
            # Structural state must change on every rank. CT registration during
            # untie is itself collective, so it cannot be inside the source branch.
            with OffloadCache.disable_onloading():
                tied = model.model.embed_tokens.weight is model.lm_head.weight
                if is_distributed():
                    targets = {
                        op.target
                        for op in (self._plan.embedding, *self._plan.rotations)
                    }
                    targets.update(f.norm for f in self._plan.fusions)
                    for name in targets:
                        module = model.get_submodule(name)
                        for key in ("weight", "bias"):
                            value = getattr(module, key, None)
                            if (
                                value is not None
                                and value in OffloadCache.keep_onloaded_values
                            ):
                                raise ValueError(
                                    "QuaRot must precede resident calibration weights"
                                )
            if tied:
                untie_word_embeddings(model)
        except Exception as caught:
            error = caught
        metadata = {
            "status": "in_progress",
            "version": 1,
            "topology": "glm_mla",
            "seed": self.seed,
            "block_size": self.block_size,
            "precision": str(self.precision).removeprefix("torch."),
        }
        model.config.quarot_config = metadata
        try:
            finish_transform_phase(error, "QuaRot preparation")
            if report_progress:
                logger.info(
                    "[QuaRot] preparation / validation complete: {:.2f} s",
                    perf_counter() - started,
                )
            error = None
            try:
                if not is_distributed() or is_source_process():
                    fusion_progress = _QuaRotProgress(
                        "norm fusion",
                        sum(len(fusion.consumers) for fusion in self._plan.fusions),
                    )
                    for fusion in self._plan.fusions:
                        fuse_norm_linears(
                            model.get_submodule(fusion.norm),
                            fusion_progress.track(
                                [model.get_submodule(name) for name in fusion.consumers]
                            ),
                            precision=self.precision,
                        )
                    fusion_progress.complete()
                    rotation_progress = _QuaRotProgress(
                        "rotation", 1 + len(self._plan.rotations)
                    )
                    for operation in rotation_progress.track(
                        (self._plan.embedding, *self._plan.rotations)
                    ):
                        self._rotate(model, operation)
                    rotation_progress.complete()
            except Exception as caught:
                error = caught
            finish_transform_phase(error, "QuaRot transform")
        except Exception:
            # A model-sized rollback copy would defeat offloading. Preflight catches
            # topology errors; unexpected I/O/OOM failures poison this model instead.
            metadata["status"] = "failed"
            self._matrices.clear()
            raise
        metadata["status"] = "applied"
        self._applied = True
        self._matrices.clear()
        if report_progress:
            logger.info("[QuaRot] total complete: {:.2f} s", perf_counter() - started)

    def on_finalize(self, state: State, **kwargs) -> bool:
        self._matrices.clear()
        self._plan = None
        self._model_ref = None
        return True
