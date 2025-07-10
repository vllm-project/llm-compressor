from typing import Iterable

import torch
from compressed_tensors import update_offload_parameter

__all__ = ["fuse_norm_linears"]


def fuse_norm_linears(norm: torch.nn.Module, linears: Iterable[torch.nn.Linear]):
    """
    Fuse a norm layer into subsequent linear layers. This useful for ensuring transform
    invariance between norm and linear layers.

    Note that a model cannot be properly trained after its norms have been fused

    :param norm: norm layer whose weight will be fused into subsequent linears
    :param linears: linear layers which directly follow the norm layer
    """
    if isinstance(norm, torch.nn.RMSNorm):
        for linear in linears:
            # spinquant does this op in float64
            new_weight = linear.weight * norm.weight
            update_offload_parameter(linear, "weight", new_weight)

        update_offload_parameter(norm, "weight", torch.ones_like(norm.weight))

    else:
        raise ValueError(f"Cannot fuse norm of type {type(norm)}")
