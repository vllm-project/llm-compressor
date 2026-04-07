import torch

from llmcompressor.utils.dev import get_main_device

__all__ = ["get_awq_precision"]


def get_awq_precision() -> torch.dtype:
    """
    Return the floating-point dtype to use for AWQ computations.

    MPS does not support float64, so fall back to float32 on that backend.
    """
    return torch.float32 if get_main_device().type == "mps" else torch.float64
