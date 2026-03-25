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
- norm_calibration_context: Context manager that applies norm conversion
"""

import contextlib
from abc import ABC, abstractmethod

import torch
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name
from loguru import logger
from transformers import PreTrainedModel

__all__ = [
    "NormCalibrationModule",
    "norm_calibration_context",
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
    alias=["Gemma2RMSNorm", "Gemma3RMSNorm", "Qwen3NextRMSNorm"],
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


@contextlib.contextmanager
def norm_calibration_context(model: PreTrainedModel):
    """
    Context manager that converts offset-norm modules to standard-norm.

    This scans all modules in the model and replaces any offset-norm modules
    (``output * (1 + weight)``) with standard-norm equivalents
    (``output * weight``).  After the context exits, modules are restored
    to their original convention with updated weights.

    The model is modified in-place, so the same model object should be used
    within the context.

    Args:
        model: The model to apply norm conversion to (modified in-place)

    Example:
        with norm_calibration_context(model):
            # Modifiers see standard norm weights
            run_calibration(model)
        # Norms restored to offset convention with smoothed weights
    """

    replaced = {}

    # Step 1: Collect all offset-norm modules that need replacement
    logger.debug("Entering norm calibration context")
    modules_to_replace = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if _is_registered(class_name, NormCalibrationModule):
            modules_to_replace.append((name, module, class_name))

    # Step 2: Replace modules
    if modules_to_replace:
        logger.info(f"Found {len(modules_to_replace)} offset-norm modules to convert")
        for name, module, class_name in modules_to_replace:
            replacement = NormCalibrationModule.load_from_registry(
                class_name,
                original=module,
                config=model.config,
            )
            model.set_submodule(name, replacement)
            replaced[name] = (module, replacement)

    try:
        yield
    finally:
        # Step 3: Restore original modules with updated weights
        if replaced:
            logger.info(f"Restoring {len(replaced)} norm modules to offset convention")
        for name, (original, replacement) in replaced.items():
            restored = replacement.restore(original)
            model.set_submodule(name, restored)


def _is_registered(name: str, subclass: RegistryMixin):
    lookup = standardize_lookup_name(name)
    return (
        lookup in subclass.registered_names() or lookup in subclass.registered_aliases()
    )
