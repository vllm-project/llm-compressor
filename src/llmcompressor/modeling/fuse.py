from typing import Iterable

import torch
from compressed_tensors.offload import update_offload_parameter

__all__ = ["center_embeddings", "fuse_norm_linears"]


PRECISION = torch.float64


def center_embeddings(embedding: torch.nn.Module):
    """
    Shift each embedding to have a mean of zero

    :param embedding: embedding module containing embeddings to center
    """
    if not hasattr(embedding, "weight"):
        raise ValueError(f"Cannot fuse norm of type {type(embedding)}")

    weight = embedding.weight.to(PRECISION)
    weight = weight - weight.mean(dim=-1, keepdim=True)
    update_offload_parameter(embedding, "weight", weight)


def fuse_norm_linears(norm: torch.nn.Module, linears: Iterable[torch.nn.Linear]):
    """
    Fuse the scaling operation of norm layer into subsequent linear layers.
    This useful for ensuring transform invariance between norm and linear layers.

    Note that unitary transforms (rotation) commute with normalization, but not scaling

    :param norm: norm layer whose weight will be fused into subsequent linears
    :param linears: linear layers which directly follow the norm layer
    """
    if not hasattr(norm, "weight"):
        raise ValueError(f"Cannot fuse norm of type {type(norm)}")

    for linear in linears:
        # NOTE: spinquant does this op in float64
        linear_weight = linear.weight.to(PRECISION) * norm.weight.to(PRECISION)
        update_offload_parameter(linear, "weight", linear_weight)

    update_offload_parameter(norm, "weight", torch.ones_like(norm.weight))
