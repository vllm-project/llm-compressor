"""
Utilities for MoNe (Mixture of Novices): expert collapse.

The lowest-saliency experts are *collapsed* into constant "novice" vectors
(each expert's mean output, an unbiased estimator of its contribution).
Retained experts are left-shifted while the router keeps all ``R + N`` rows,
so novices continue to receive their original router weights at inference
time.
"""

import torch
import torch.nn as nn
from compressed_tensors.offload import get_execution_device

__all__ = ["set_novices"]


def set_novices(experts: nn.Module, novices: torch.Tensor) -> None:
    exec_device = get_execution_device(experts)
    target_dtype = novices.dtype
    for param in experts.parameters():
        if param.is_floating_point():
            target_dtype = param.dtype
            break
    novices = novices.to(device=exec_device, dtype=target_dtype).contiguous()
    experts.register_parameter("novices", nn.Parameter(novices, requires_grad=False))
