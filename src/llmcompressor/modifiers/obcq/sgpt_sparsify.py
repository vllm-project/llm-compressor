import math
from typing import Dict, Optional, Tuple

import torch
import transformers
from loguru import logger

SGPT_PRECISION = torch.float32


def make_empty_hessian(
    module: torch.nn.Module, device: Optional[torch.device] = None
) -> torch.Tensor:
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros((num_columns, num_columns), device=device, dtype=SGPT_PRECISION)


def accumulate_hessian(
    inp: torch.Tensor,
    module: torch.nn.Module,
    H: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, int]:
    inp = inp.to(device=H.device)
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)

    num_added = inp.shape[0]  # note this is the number of dataset samples, not
    # multiplied by the sequence length

    if isinstance(module, (torch.nn.Linear, transformers.Conv1D)):
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

    if isinstance(module, torch.nn.Conv2d):
        unfold = torch.nn.Unfold(
            module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        inp = unfold(inp)
        inp = inp.permute([1, 0, 2])
        inp = inp.flatten(1)

    H *= num_samples / (num_samples + num_added)
    num_samples += num_added

    inp = inp.to(dtype=SGPT_PRECISION)
    inp = math.sqrt(2 / num_samples) * inp
    H += inp.matmul(inp.t())

    return H, num_samples


def sparsify_weight(
    module: torch.nn.Module,
    hessians_dict: Dict[torch.nn.Module, torch.Tensor],
    sparsity: float,
    prune_n: int,
    prune_m: int,
    block_size: int,
    dampening_frac: float,
    preserve_sparsity_mask: bool,
) -> torch.Tensor:
    """
    Run pruning on the layer up to the target sparsity value.

    :param module: module with weight being sparsified
    :param hessian_dict: dictionary containing preaccumulated hessian for sparsification
    :param sparsity: target sparsity to reach for layer
    :param prune_n: N for N:M pruning
    :param prune_m: M for N:M pruning
    :param block_size: Number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param preserve_sparsity_mask: Extend or ignore the base sparsity mask
    """
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()
    H = hessians_dict[module]  # unfortunately python does not have a `move` keyword
    del hessians_dict[module]  # so we have to delete the original reference manually

    # standardize shape and dtype
    if isinstance(module, torch.nn.Conv2d):
        W = W.flatten(1)
    elif isinstance(module, transformers.Conv1D):
        W.transpose_(0, 1)
    W = W.to(dtype=SGPT_PRECISION)
    num_rows = W.shape[0]
    num_columns = W.shape[1]

    # mask dead hessian values
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # compute inverse hessian in place to save memory
    try:
        damp = dampening_frac * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
    except torch._C._LinAlgError:
        logger.warning(
            "Failed to invert hessian due to numerical instability. Consider "
            "increasing SparseGPTModifier.dampening_frac, increasing the number "
            "of calibration samples, or shuffling the calibration dataset"
        )
        Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)

    # sparsity mask
    # TODO: consider computing sparsity mask in the same way and place as gptq
    mask = None
    if preserve_sparsity_mask:
        # compute existing sparsity mask
        mask = torch.where(
            W == 0,
            torch.tensor(1, dtype=torch.bool),
            torch.tensor(0, dtype=torch.bool),
        )
        current_sparsity = mask.sum() / W.numel()
        if current_sparsity > sparsity:
            raise ValueError(
                "The target sparsity is lower than the sparsity "
                "of the base model. Please retry "
                "after turning preserve_sparsity_mask=False"
            )

    losses = torch.zeros(num_rows, device=module.weight.device)

    # See section 3.4 of https://arxiv.org/abs/2203.07259
    for i1 in range(0, num_columns, block_size):
        i2 = min(i1 + block_size, num_columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if prune_n == 0:
            if mask is not None:
                mask1 = mask[:, i1:i2]
                if int(W1.numel() * sparsity) > mask1.sum():
                    # target sparsity is higher than base sparsity, extend mask1
                    tmp = (
                        (~mask[:, i1:i2])
                        * W1**2
                        / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    )
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
        else:
            if mask is not None:
                mask1 = mask[:, i1:i2]
            else:
                mask1 = torch.zeros_like(W1) == 1

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if prune_n != 0 and i % prune_m == 0:
                tmp = (
                    W1[:, i : (i + prune_m)] ** 2
                    / (torch.diag(Hinv1)[i : (i + prune_m)].reshape((1, -1))) ** 2
                )
                if mask is not None:
                    tmp = tmp * (~mask[:, i : (i + prune_m)])

                mask1.scatter_(
                    1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True
                )

            q = w.clone()
            q[mask1[:, i]] = 0

            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        losses += torch.sum(Losses1, 1) / 2

        if preserve_sparsity_mask:
            # respect the sparsity of other groups
            # really not needed, but kept for explicitness
            W[:, i2:] -= (~mask[:, i2:]) * Err1.matmul(Hinv[i1:i2, i2:])
        else:
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    if isinstance(module, transformers.Conv1D):
        W.transpose_(0, 1)
    W = W.reshape(final_shape).to(final_dtype)

    loss = torch.sum(losses).item()
    return loss, W
