"""
PyTorch-specific helper functions for model compression.

Provides utility functions for PyTorch model operations including
batch processing, padding mask application, and model architecture
detection. Supports MoE (Mixture of Experts) models and specialized
tensor operations for compression workflows.
"""

from typing import Dict, TYPE_CHECKING

import torch
from torch.nn import Module

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier

__all__ = [
    "apply_pad_mask_to_batch",
    "is_moe_model",
]


def apply_pad_mask_to_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Apply a mask to the input ids of a batch. This is used to zero out
    padding tokens so they do not contribute to the hessian calculation in the
    GPTQ and SparseGPT algorithms

    Assumes that `attention_mask` only contains zeros and ones

    :param batch: batch to apply padding to if it exists
    :return: batch with padding zeroed out in the input_ids
    """
    if "attention_mask" in batch:
        for key in ("input_ids", "decoder_input_ids"):
            if key in batch:
                batch[key] = batch[key] * batch["attention_mask"]

    return batch


def is_moe_model(model: Module) -> bool:
    """
    Check if the model is a mixture of experts model

    :param model: the model to check
    :return: True if the model is a mixture of experts model
    """

    # Check for MoE components
    for _, module in model.named_modules():
        module_name = module.__class__.__name__
        if "MoE" in module_name or "Expert" in module_name:
            return True

    # Check config for MoE attributes
    if hasattr(model, "config"):
        if any(
            "moe" in attr.lower() or "expert" in attr.lower()
            for attr in dir(model.config)
        ):
            return True

    return False


def get_modifier_targets(model: torch.nn.Module, modifier: "Modifier") -> list[torch.nn.Module]:
    """
    Get all modules which will be modified by the given modifier
    """
    from llmcompressor.modifiers.quantization import QuantizationMixin
    from llmcompressor.modifiers.transform import SpinQuantModifier

    for modifier in modifiers:
        if not modifier.initialized:
            # TODO
            pass

        if isinstance(modifier, QuantizationMixin):
            modifier.resolved_targets
        elif isinstance(modifier, SpinQuantModifier):
            modifier.mappings.
            
        else:
            raise NotImplementedError(f"Unsupported modifier {modifier.__class__}")