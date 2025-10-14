"""
Simplified interface for MoE model calibration.

MoE (Mixture of Experts) models route tokens to different expert networks.
During calibration for quantization/compression, we need to ensure ALL experts
see data, not just the ones selected by the router. This module provides the
infrastructure to temporarily modify MoE modules for proper calibration.

Key components:
- MoECalibrationModule: Abstract base class for calibration modules
- MOE_CALIBRATION_MODULES: Registry mapping module class names to calibration classes
- moe_calibration_context: Context manager that applies calibration to a model
"""

import contextlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import torch
from loguru import logger
from transformers import PreTrainedModel

__all__ = [
    "MoECalibrationModule",
    "MOE_CALIBRATION_MODULES",
    "register_moe_calibration",
    "moe_calibration_context",
]


class MoECalibrationModule(ABC, torch.nn.Module):
    """
    Abstract base class for MoE calibration modules.
    
    Calibration modules replace original MoE modules during the calibration
    phase to ensure all experts receive data for proper quantization statistics.
    
    Subclasses must:
    1. Implement `from_original()` to create calibration module from original
    2. Set `is_permanent` to indicate if module should stay in calibration form
    3. Optionally implement `restore()` if is_permanent=False
    """

    is_permanent: bool = False

    @classmethod
    @abstractmethod
    def from_original(
        cls,
        original: torch.nn.Module,
        config: Any,
        calibrate_all_experts: bool = True,
    ) -> "MoECalibrationModule":
        """
        Create a calibration module from the original MoE module.

        Args:
            original: The original MoE module to convert
            config: Model configuration (contains num_experts, etc.)
            calibrate_all_experts: If True, send all tokens to all experts.
                                   If False, use normal routing.

        Returns:
            Instance of the calibration module
        """
        pass

    def restore(self) -> torch.nn.Module:
        """
        Restore the original module structure.
        
        Only needed if is_permanent=False. For permanent modules, this is a no-op.
        
        Returns:
            The original module (or self if permanent)
        """
        if self.is_permanent:
            return self
        raise NotImplementedError(
            f"{self.__class__.__name__} has is_permanent=False but doesn't "
            "implement restore()"
        )


# Registry: module class name -> calibration module class
MOE_CALIBRATION_MODULES: Dict[str, Type[MoECalibrationModule]] = {}


def register_moe_calibration(module_class_name: str):
    """
    Decorator to register a MoE calibration module.
    
    Usage:
        @register_moe_calibration("DeepseekV3MoE")
        class CalibrationDeepseekV3MoE(MoECalibrationModule):
            ...
    
    Args:
        module_class_name: The class name of the original module to replace
    """

    def decorator(cls: Type[MoECalibrationModule]) -> Type[MoECalibrationModule]:
        if not issubclass(cls, MoECalibrationModule):
            raise TypeError(
                f"{cls.__name__} must inherit from MoECalibrationModule"
            )
        MOE_CALIBRATION_MODULES[module_class_name] = cls
        return cls

    return decorator


@contextlib.contextmanager
def moe_calibration_context(
    model: PreTrainedModel,
    calibrate_all_experts: bool = True,
):
    """
    Context manager that applies MoE calibration to a model.
    
    This scans all modules in the model and replaces any MoE modules with their
    calibration equivalents. After the context exits, non-permanent modules are
    restored to their original form.
    
    Args:
        model: The model to apply MoE calibration to
        calibrate_all_experts: If True, all experts see all tokens during calibration.
                               If False, use normal routing (useful for some techniques)
    
    Yields:
        The model with MoE calibration applied
        
    Example:
        with moe_calibration_context(model):
            # Run calibration - all experts will see data
            for batch in dataloader:
                model(**batch)
        # Model is now restored (unless permanent)
    """
    replaced = {}

    # Step 1: Find and replace MoE modules
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name in MOE_CALIBRATION_MODULES:
            calibration_cls = MOE_CALIBRATION_MODULES[class_name]
            replacement = calibration_cls.from_original(
                module,
                model.config,
                calibrate_all_experts,
            )
            model.set_submodule(name, replacement)
            replaced[name] = (module, replacement)

    # Log what was replaced
    if replaced:
        logger.info(f"Replaced {len(replaced)} MoE modules for calibration")
        permanent_count = sum(
            1 for _, (_, repl) in replaced.items() if repl.is_permanent
        )
        if permanent_count > 0:
            logger.info(
                f"{permanent_count}/{len(replaced)} modules will remain in "
                "calibration form (permanent)"
            )
        if permanent_count < len(replaced):
            logger.info(
                f"{len(replaced) - permanent_count}/{len(replaced)} modules will "
                "be restored after calibration"
            )

    try:
        yield model
    finally:
        # Step 2: Restore non-permanent modules
        for name, (original, replacement) in replaced.items():
            if not replacement.is_permanent:
                restored = replacement.restore()
                model.set_submodule(name, restored)
