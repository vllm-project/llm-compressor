from typing import Dict

import torch
from torch.nn import Module

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
