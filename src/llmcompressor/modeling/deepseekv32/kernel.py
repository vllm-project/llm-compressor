# This file contains code copied from the DeepSeek-V3.2-Exp project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2025 DeepSeek


import torch


def bf16_index(
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using bfloat16 precision.

    Computes: output[b,m,n] = sum_over_h(ReLU(K[b,n,:] · Q[b,m,h,:]^T))

    Args:
        q (torch.Tensor): Query tensor of shape (b, m, h, d)
        k (torch.Tensor): Key tensor of shape (b, n, d)

    Returns:
        torch.Tensor: Output tensor of shape (b, m, n)
    """
    # Use einsum for memory-efficient computation
    # k: (b, n, d), q: (b, m, h, d)
    # logits[b, m, n, h] = sum_d(k[b, n, d] * q[b, m, h, d])
    logits = torch.einsum("bnd,bmhd->bmnh", k, q)  # (b, m, n, h)

    # Apply ReLU
    logits = torch.relu(logits)

    # Sum over heads to get (b, m, n)
    return logits.sum(dim=-1).to(torch.float32)
