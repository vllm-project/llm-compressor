from itertools import cycle
from typing import Callable, Dict, Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from llmcompressor.pytorch.utils import tensors_module_forward, tensors_to_device

__all__ = [
    "apply_pad_mask_to_batch",
    "run_calibration_forward",
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


def run_calibration_forward(
    model: Module,
    calibration_dataloader: DataLoader,
    num_calibration_steps: Optional[int] = None,
    calibration_function: Optional[Callable] = None,
    device: Optional[str] = None,
    mask_padding: bool = False,
):
    """
    Helper function used by one-shot modifiers, runs calibration data through a model to
    update modifier statistics and trigger hooks

    :param model: PyTorch model to run
    :param calibration_dataloader: data to use for calibration
    :param num_calibration_steps: number of items in calibration_dataloader to process,
        None or a negative number to process all available data
    :param calibration_function: option to pass a custom forward function for model
    :param device: option to move the model to a specific device before calibration
    :param mask_padding: whether to zero out padding tokens during calibration
    """
    model.eval()

    forward_fn: Callable = (
        calibration_function if calibration_function else tensors_module_forward
    )

    # move model to optional specified device if it is not already there
    model_device = next(model.parameters()).device
    if device is not None and model_device != device:
        model.to(device)
        model_device = next(model.parameters()).device
    _dataloader = (
        calibration_dataloader
        if num_calibration_steps is None
        else cycle(calibration_dataloader)
    )

    # run through the calibration data
    for batch_idx, batch in enumerate(tqdm(_dataloader)):
        if num_calibration_steps and batch_idx >= num_calibration_steps:
            break
        if mask_padding:
            batch = apply_pad_mask_to_batch(batch)
        batch = tensors_to_device(batch, model_device)
        with torch.no_grad():
            forward_fn(batch, module=model)


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
