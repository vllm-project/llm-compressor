"""
Simplified interface for MoE model calibration.

MoE (Mixture of Experts) models route tokens to different expert networks.
During calibration for quantization/compression, we need to ensure ALL experts
see data, not just the ones selected by the router. This module provides the
infrastructure to temporarily modify MoE modules for proper calibration.

Key components:
- MoECalibrationModule: Abstract base class for calibration modules
- moe_calibration_context: Context manager that applies calibration to a model
"""

import contextlib
from abc import ABC

import torch
import torch.distributed as dist
from compressed_tensors.offload import is_distributed
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name
from loguru import logger
from tqdm import tqdm
from transformers import PreTrainedModel

__all__ = [
    "MoECalibrationModule",
    "moe_calibration_context",
]


class MoECalibrationModule(ABC, torch.nn.Module, RegistryMixin):
    """
    Abstract base class for MoE calibration modules.

    Calibration modules replace original MoE modules during the calibration
    phase to ensure all experts receive data for proper quantization statistics.

    Subclasses must:
    1. Implement `__init__()` with signature:
       (self, original, config, calibrate_all_experts=True)
    2. Set `is_permanent` to indicate if module should stay in calibration form
    3. Optionally implement `restore()` if is_permanent=False
    """

    is_permanent: bool = False

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
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

    The model is modified in-place, so the same model object should be used
    within the context.

    Args:
        model: The model to apply MoE calibration to (modified in-place)
        calibrate_all_experts: If True, all experts see all tokens during calibration.
                               If False, use normal routing (useful for some techniques)

    Example:
        with moe_calibration_context(model):
            # Run calibration - all experts will see data
            for batch in dataloader:
                model(**batch)
        # Model is now restored (unless permanent)
    """

    replaced = {}

    # Step 1: Collect all MoE modules that need replacement
    logger.debug("Entering MoE calibration context")
    modules_to_replace = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if _is_registered(class_name, MoECalibrationModule):
            modules_to_replace.append((name, module, class_name))

    # Step 2: Replace modules with progress bar
    if modules_to_replace:
        logger.info(f"Found {len(modules_to_replace)} MoE modules to replace")
        for name, module, class_name in tqdm(
            modules_to_replace, desc="Replacing MoE modules for calibration"
        ):
            replacement = MoECalibrationModule.load_from_registry(
                class_name,
                original=module,
                config=model.config,
                calibrate_all_experts=calibrate_all_experts,
            )
            model.set_submodule(name, replacement)
            replaced[name] = (module, replacement)
            if is_distributed():
                dist.barrier()

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
        yield
    finally:
        # Step 2: Restore non-permanent modules
        for name, (original, replacement) in replaced.items():
            if not replacement.is_permanent:
                restored = replacement.restore(original)
                model.set_submodule(name, restored)


def _is_registered(name: str, subclass: RegistryMixin):
    return standardize_lookup_name(name) in subclass.registered_names()
