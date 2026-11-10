"""
Lazy calibration transformations for offset-norm layers.

Some architectures (Gemma, Qwen3Next) use an offset normalization pattern where
the forward pass computes ``output * (1 + weight)``. This breaks modifiers that
smooth norm weights because they update the stored weight as though the forward
pass computed ``output * weight``.

This module records transformations for supported norm types without reading or
copying their weights. The sequential pipeline applies a transformation after its
subgraph is onloaded and restores the offset convention before recompression or
offload.
"""

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch
from compressed_tensors.offload import OffloadCache, update_offload_parameter
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name
from torch.nn import Module
from transformers import PreTrainedModel

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "CalibrationOffsetNorm",
    "NormCalibrationModule",
    "apply_norm_calibration",
    "restore_norm_calibration",
    "norm_calibration_context",
]


class NormCalibrationModule(ABC, RegistryMixin):
    """Transformation for one norm module during calibration."""

    is_permanent: bool = False

    def __init__(self, original: Module):
        self.original = original
        self._original_dtype: torch.dtype | None = None
        self._original_forward = None
        self._forward_is_offload_wrapped = False
        self._active = False

    @abstractmethod
    def apply(self) -> None:
        """Convert the module to standard ``output * weight`` semantics."""
        raise NotImplementedError()

    @abstractmethod
    def restore(self) -> None:
        """Restore the offset convention after calibration transformations."""
        raise NotImplementedError


@NormCalibrationModule.register(
    "GemmaRMSNorm",
    alias=[
        "Gemma2RMSNorm",
        "Gemma3RMSNorm",
        "MuseGlimmerTextCenteredRMSNorm",
        "Qwen3NextRMSNorm",
        "Qwen3_5RMSNorm",
        "Qwen3_5MoeRMSNorm",
        "VaultGemmaRMSNorm",
        "RecurrentGemmaRMSNorm",
        "T5GemmaRMSNorm",
        "T5Gemma2RMSNorm",
        "Step3p7RMSNorm",
        "MiniMaxM3VLRMSNorm",
        "MuseSparkTextCenteredRMSNorm",
    ],
)
class CalibrationOffsetNorm(NormCalibrationModule):
    """Temporarily represents an offset norm with a standard RMSNorm weight."""

    def apply(self) -> None:
        if self._active:
            return

        module = self.original
        self._original_dtype = module.weight.dtype
        self._forward_is_offload_wrapped = isinstance(
            module._parameters, OffloadCache
        ) and hasattr(module, "_original_forward_func")
        if self._forward_is_offload_wrapped:
            self._original_forward = module._original_forward_func
        else:
            self._original_forward = module.forward

        self._active = True
        weight = module.weight.detach().float()
        update_offload_parameter(
            module,
            "weight",
            (weight + 1.0).to(self._original_dtype),
        )
        if self._forward_is_offload_wrapped:
            module._original_forward_func = _standard_rms_norm_forward
        else:
            module.forward = _standard_rms_norm_forward.__get__(module, type(module))

    def restore(self) -> None:
        if not self._active:
            return

        module = self.original
        weight = module.weight.detach().float()
        update_offload_parameter(
            module,
            "weight",
            (weight - 1.0).to(self._original_dtype),
        )
        if self._forward_is_offload_wrapped:
            module._original_forward_func = self._original_forward
        else:
            module.forward = self._original_forward
        self._original_dtype = None
        self._original_forward = None
        self._forward_is_offload_wrapped = False
        self._active = False


def apply_norm_calibration(modules: Iterable[Module]) -> list[NormCalibrationModule]:
    """Apply registered norm transformations to modules in an onloaded scope.

    Returns the transformations that were applied. Keep this list until the scope
    finishes and pass it to :func:`restore_norm_calibration`.
    """
    transformations = []
    for module in modules:
        class_name = module.__class__.__name__
        if not _is_registered(class_name, NormCalibrationModule):
            continue

        transformation = NormCalibrationModule.load_from_registry(
            class_name,
            original=module,
        )
        transformation.apply()
        transformations.append(transformation)

    return transformations


def restore_norm_calibration(
    transformations: Sequence[NormCalibrationModule],
) -> None:
    """Restore transformations applied to a completed scope."""
    for transformation in reversed(transformations):
        transformation.restore()


@contextlib.contextmanager
def norm_calibration_context(model: PreTrainedModel) -> "Iterator[None]":
    """Convert all registered offset norms for the duration of a context.

    :param model: model whose offset-norm modules need calibration transforms
    """
    transformations = []
    try:
        transformations = apply_norm_calibration(model.modules())
        yield
    finally:
        restore_norm_calibration(transformations)


def _standard_rms_norm_forward(module: Module, x: torch.Tensor) -> torch.Tensor:
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + module.eps)
    output = output * module.weight.float()
    return output.type_as(x)


def _is_registered(name: str, subclass: RegistryMixin) -> bool:
    lookup = standardize_lookup_name(name)
    return (
        lookup in subclass.registered_names() or lookup in subclass.registered_aliases()
    )
