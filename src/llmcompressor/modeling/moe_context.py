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
import os
from abc import ABC

import torch
import torch.distributed as dist
from compressed_tensors.offload import get_execution_device, get_offloaded_device, is_distributed
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import offload_module
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name
from loguru import logger
from tqdm import tqdm
from transformers import PreTrainedModel

__all__ = [
    "MoECalibrationModule",
    "moe_calibration_context",
]


def _log_memory_stats(prefix: str):
    """Log mmap count and memory usage for debugging."""
    try:
        # Get mmap count for current process
        pid = os.getpid()
        with open(f"/proc/{pid}/maps", "r") as f:
            mmap_count = sum(1 for _ in f)

        # Get shared memory usage
        shm_usage = 0
        if os.path.exists("/dev/shm"):
            for entry in os.listdir("/dev/shm"):
                if entry.startswith("torch_"):
                    path = os.path.join("/dev/shm", entry)
                    if os.path.isfile(path):
                        shm_usage += os.path.getsize(path)

        # Get max_map_count limit
        with open("/proc/sys/vm/max_map_count", "r") as f:
            max_map_count = int(f.read().strip())

        logger.info(
            f"{prefix} | mmaps: {mmap_count}/{max_map_count} "
            f"({100*mmap_count/max_map_count:.1f}%) | "
            f"/dev/shm usage: {shm_usage/(1024**3):.2f} GB"
        )
    except Exception as e:
        logger.warning(f"Failed to log memory stats: {e}")


def _apply_offloading_to_replacement(original: torch.nn.Module, replacement: torch.nn.Module):
    """
    Apply the same offloading configuration from original to replacement module.

    If the original module uses OffloadCache, this recursively applies the same
    offloading settings to all submodules of the replacement that have parameters.
    """
    if not isinstance(original._parameters, OffloadCache):
        return  # Original doesn't use offloading, nothing to do

    # Get offloading settings from original
    onload_device = get_execution_device(original)
    offload_device = get_offloaded_device(original)

    # Get additional kwargs if using disk cache
    offload_kwargs = {}
    if hasattr(original._parameters, 'offload_dir'):
        offload_kwargs['offload_dir'] = original._parameters.offload_dir

    logger.debug(
        f"Applying offloading to replacement module: "
        f"onload={onload_device}, offload={offload_device}"
    )

    # Apply offloading to all submodules that have parameters
    offloaded_count = 0
    for module in replacement.modules():
        # Only offload modules that have parameters and aren't already offloaded
        if (len(list(module.parameters(recurse=False))) > 0 and
            not isinstance(module._parameters, OffloadCache)):
            offload_module(module, onload_device, offload_device, **offload_kwargs)
            offloaded_count += 1

    logger.debug(f"Offloaded {offloaded_count} submodules in replacement")


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
        _log_memory_stats("Before MoE replacement")

        for idx, (name, module, class_name) in enumerate(
            tqdm(modules_to_replace, desc="Replacing MoE modules for calibration")
        ):
            replacement = MoECalibrationModule.load_from_registry(
                class_name,
                original=module,
                config=model.config,
                calibrate_all_experts=calibrate_all_experts,
            )

            # Apply the same offloading settings as the original module
            _apply_offloading_to_replacement(module, replacement)

            model.set_submodule(name, replacement)

            # Only store original if we need to restore it later
            if replacement.is_permanent:
                replaced[name] = (None, replacement)
                del module  # Help GC by explicitly deleting
            else:
                replaced[name] = (module, replacement)

            # Log every 10 replacements or on the last one
            if (idx + 1) % 10 == 0 or idx == len(modules_to_replace) - 1:
                _log_memory_stats(f"After replacing {idx+1}/{len(modules_to_replace)} modules")

            if is_distributed():
                dist.barrier()

        _log_memory_stats("After all MoE replacements")

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
