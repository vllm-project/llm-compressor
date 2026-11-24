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
from compressed_tensors.offload.module import remove_module_offload
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.modeling.module_restructure import (
    ModuleRestructure,
    copy_module_state,
    replace_module,
)

__all__ = [
    "NormCalibrationModule",
    "get_norm_calibration_targets",
    "norm_calibration_context",
    "OffsetNormRestructure",
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

    def __init__(self, original: torch.nn.Module):
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


class OffsetNormRestructure(ModuleRestructure):
    """Own offset-norm conversion, materialization, and restoration."""

    def apply(
        self,
        model: torch.nn.Module,
        mapping: dict[torch.nn.Module, torch.nn.Module] | None = None,
    ) -> torch.nn.Module:
        if mapping is not None:
            self.delayed = mapping
        targets = get_norm_calibration_targets(model)
        for name, source, class_name in targets:
            target = NormCalibrationModule.load_from_registry(
                class_name, original=source
            )
            if mapping is not None:
                original = mapping.pop(source)
                self.mapping[original] = target
                mapping[target] = original
            model = replace_module(model, source, target)
        return model

    def restructure_sources(self, model, sources, device, offload_kwargs):
        for source in sources.values():
            target = self.mapping.get(source)
            if target is not None and target in self.delayed:
                converted = type(target)(original=source)
                copy_module_state(target, converted, device)
                self.delayed.pop(target)

    def restore(self, model, modules=None):
        for source, target in list(self.mapping.items()):
            if target in self.delayed:
                continue
            if modules is not None and target not in modules.values():
                continue
            remove_module_offload(target, onload_tensors=True)
            remove_module_offload(source, onload_tensors=True)
            source.forward = source.forward.__func__.__get__(source)
            target.restore(source)
            model = replace_module(model, target, source)
            if modules is not None:
                for name, module in modules.items():
                    if module is target:
                        modules[name] = source
            del self.mapping[source]
        return model


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
    modules_to_replace = get_norm_calibration_targets(model)

    # Step 2: Replace modules
    if modules_to_replace:
        logger.info(f"Found {len(modules_to_replace)} offset-norm modules to convert")
        for name, module, class_name in modules_to_replace:
            replacement = NormCalibrationModule.load_from_registry(
                class_name,
                original=module,
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


def get_norm_calibration_targets(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module, str]]:
    """Return modules converted by :func:`norm_calibration_context`."""
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
