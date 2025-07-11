from typing import Iterable

import torch
from compressed_tensors import get_execution_device, align_module_device, update_offload_parameter

from transformers.models.llama.modeling_llama import LlamaRMSNorm

__all__ = ["fuse_norm_linears"]


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
            with align_module_device(norm, exec_device), align_module_device(linear, exec_device):
                
                weight_dtype = linear.weight.dtype

                new_weight = linear.weight.to(torch.float64) * norm.weight.to(torch.float64)

                new_weight = new_weight.to(weight_dtype)
            
            update_offload_parameter(linear, "weight", new_weight)

        update_offload_parameter(norm, "weight", torch.ones_like(norm.weight))

    else:
        raise ValueError(f"Cannot fuse norm of type {type(norm)}")
