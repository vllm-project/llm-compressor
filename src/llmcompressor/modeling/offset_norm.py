"""
Calibration context for offset-norm layers.

Some architectures (Gemma, Qwen3Next) use an offset normalization pattern where
the forward pass computes ``output * (1 + weight)`` instead of the standard
``output * weight``.  This breaks any modifier that smooths norm weights
(AWQ, SmoothQuant, SpinQuant) because dividing a (1+weight) parameter by scales
produces incorrect results.

This module provides the infrastructure to temporarily replace offset-norm
modules with standard-norm equivalents during calibration, and restore the
original convention after modifiers have run.

Key components:
- NormCalibrationModule: Abstract base class for norm calibration modules
"""

from abc import ABC, abstractmethod

import torch
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name

__all__ = [
    "NormCalibrationModule",
    "get_norm_calibration_targets",
]


class NormCalibrationModule(ABC, torch.nn.Module, RegistryMixin):
    """
    Abstract base class for norm calibration modules.

    Calibration modules replace original norm modules during the calibration
    phase so that modifiers see standard ``output * weight`` semantics.
    """

    is_permanent: bool = False

    @abstractmethod
    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        """
        Restore the original module with updated weights.

        Returns:
            The original module with weights converted back to offset convention
        """
        ...


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
    """
    Replaces offset-norm modules (``output * (1 + weight)``) with standard-norm
    equivalents (``output * weight``) during calibration.

    On enter: ``self.weight = 1 + original.weight``
    On restore: ``original.weight = self.weight - 1``
    """

    is_permanent = False

    def __init__(self, original: torch.nn.Module, config):
        super().__init__()
        self.eps = original.eps
        self._orig_dtype = original.weight.dtype
        self.weight = torch.nn.Parameter(
            (1.0 + original.weight.data.float()).to(original.weight.dtype)
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * self.weight.float()
        return output.type_as(x)

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        original.weight.data = (self.weight.data.float() - 1.0).to(self._orig_dtype)
        return original


def get_norm_calibration_targets(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module, str]]:
    """Return offset-norm modules with registered calibration equivalents."""
    return [
        (name, module, module.__class__.__name__)
        for name, module in model.named_modules()
        if _is_registered(module.__class__.__name__, NormCalibrationModule)
    ]


def _is_registered(name: str, subclass: RegistryMixin):
    lookup = standardize_lookup_name(name)
    return (
        lookup in subclass.registered_names() or lookup in subclass.registered_aliases()
    )
