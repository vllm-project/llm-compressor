from typing import Iterable

import torch
from compressed_tensors import (
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm

__all__ = ["normalize_embedding", "fuse_norm_linears"]


PRECISION = torch.float64


def normalize_embedding(embedding: torch.nn.Module):
    if isinstance(embedding, (torch.nn.Embedding)):
        with align_module_device(embedding):
            weight_dtype = embedding.weight.dtype
            weight = embedding.weight.to(PRECISION)
            new_weight = weight - weight.mean(dim=-1, keepdim=True)
            new_weight = new_weight.to(weight_dtype)

        update_offload_parameter(embedding, "weight", new_weight)

    else:
        raise ValueError(f"Cannot normalize embedding of type {type(embedding)}")


def fuse_norm_linears(norm: torch.nn.Module, linears: Iterable[torch.nn.Linear]):
    """
    Fuse a norm layer into subsequent linear layers. This useful for ensuring transform
    invariance between norm and linear layers.

    Note that a model cannot be properly trained after its norms have been fused

    :param norm: norm layer whose weight will be fused into subsequent linears
    :param linears: linear layers which directly follow the norm layer
    """
    if isinstance(norm, (torch.nn.RMSNorm, LlamaRMSNorm)):
        for linear in linears:
            # NOTE: spinquant does this op in float64
            exec_device = get_execution_device(norm)
            with align_module_device(norm, exec_device), align_module_device(
                linear, exec_device
            ):
                weight_dtype = linear.weight.dtype
                new_weight = linear.weight.to(PRECISION) * norm.weight.to(PRECISION)
                new_weight = new_weight.to(weight_dtype)

            update_offload_parameter(linear, "weight", new_weight)

        update_offload_parameter(norm, "weight", torch.ones_like(norm.weight))

    else:
        raise ValueError(f"Cannot fuse norm of type {type(norm)}")
