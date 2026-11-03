"""
PyTorch-specific helper functions for model compression.

Provides utility functions for PyTorch model operations including
batch processing, padding and loss mask application, and model
architecture detection. Supports MoE (Mixture of Experts) models and
specialized tensor operations for compression workflows.
"""

import torch
from torch.nn import Module

from llmcompressor.core import active_session

__all__ = [
    "apply_pad_mask_to_batch",
    "get_current_loss_mask",
    "apply_loss_mask",
    "is_moe_model",
]


def apply_pad_mask_to_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Apply a mask to the input ids of a batch. This is used to zero out
    padding tokens so they do not contribute to the hessian calculation in the
    GPTQ algorithm

    Assumes that `attention_mask` only contains zeros and ones

    :param batch: batch to apply padding to if it exists
    :return: batch with padding zeroed out in the input_ids
    """
    if "attention_mask" in batch:
        for key in ("input_ids", "decoder_input_ids"):
            if key in batch:
                batch[key] = batch[key] * batch["attention_mask"]

    return batch


def get_current_loss_mask() -> torch.Tensor | None:
    """
    Return the loss mask of the calibration batch currently being processed, or
    None if token masking is disabled (``use_loss_mask=False``) or no batch is
    active.

    The calibration pipelines populate ``state.loss_masks`` from the ``loss_mask``
    field of the dataset and set ``state.current_batch_idx`` before running each
    batch through the model, so calibration hooks can look up the mask of the
    batch which produced their inputs.

    :return: ``[batch, seq_len]`` mask of the current batch, or None
    """
    state = active_session().state
    if state is None or not state.loss_masks:
        return None
    if not 0 <= state.current_batch_idx < len(state.loss_masks):
        return None
    return state.loss_masks[state.current_batch_idx]


def apply_loss_mask(
    activations: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """
    Flatten per-token activations into ``[num_tokens, hidden]`` and, if a loss
    mask is given, keep only the tokens where the mask is nonzero.

    This is the shared masking logic for calibration statistics which should only
    reflect a subset of the calibration tokens (e.g. assistant responses), such as
    AWQ's activation means and GPTQ's Hessian.

    :param activations: tensor whose leading dimensions index tokens, e.g.
        ``[batch, seq_len, hidden]``
    :param mask: optional mask with one entry per token, e.g. ``[batch, seq_len]``.
        None keeps every token
    :return: ``[num_kept_tokens, hidden]`` tensor
    """
    flat = activations.flatten(0, -2)
    if mask is None:
        return flat

    if mask.numel() != flat.shape[0]:
        raise ValueError(
            f"Loss mask of shape {tuple(mask.shape)} cannot be aligned with "
            f"activations of shape {tuple(activations.shape)}. Token masking is only "
            "supported for modules which see every token of the batch (e.g. not "
            "routed MoE experts)"
        )

    return flat[mask.reshape(-1).to(device=flat.device, dtype=torch.bool)]


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
    # NOTE: transformers v5 PreTrainedConfig adds property `_experts_implementation`
    # to all model configs. Skipping all attributes starting with `_` to avoid this.
    if hasattr(model, "config"):
        if any(
            "moe" in attr.lower() or "expert" in attr.lower()
            for attr in dir(model.config)
            if not attr.startswith("_")
        ):
            return True

    return False
