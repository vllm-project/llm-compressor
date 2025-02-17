from typing import Dict, Optional

import torch
import transformers

WANDA_PRECISION = torch.float32


def make_empty_row_scalars(
    module: torch.nn.Module, device: Optional[torch.device] = None
) -> torch.Tensor:
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros(num_columns, device=device)


def accumulate_row_scalars(
    inp: torch.Tensor,
    module: torch.nn.Module,
    row_scalars: torch.Tensor,
    num_samples: int,
):
    inp = inp.to(device=row_scalars.device)
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

    row_scalars *= num_samples / (num_samples + num_added)
    num_samples += num_added

    inp = inp.type(WANDA_PRECISION)
    row_scalars += torch.norm(inp, p=2, dim=1) ** 2 / num_samples

    return row_scalars, num_samples


def sparsify_weight(
    module: torch.nn.Module,
    row_scalars_dict: Dict[torch.nn.Module, torch.Tensor],
    sparsity: float,
    prune_n: int,
    prune_m: int,
) -> torch.Tensor:
    """
    Run pruning on the layer up to the target sparsity value.

    :param sparsity: target sparsity to reach for layer
    :param prunen: N for N:M pruning
    :param prunem: M for N:M pruning
    """
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.data.clone()
    if isinstance(module, torch.nn.Conv2d):
        W = W.flatten(1)
    if isinstance(module, transformers.Conv1D):
        W = W.t()
    W = W.to(dtype=WANDA_PRECISION)
    S = row_scalars_dict[module]  # unfortunately python does not have a `move` keyword
    del row_scalars_dict[module]  # so we have to delete the original reference manually

    W_metric = torch.abs(W) * torch.sqrt(S.reshape((1, -1)))

    # initialize a mask to be all False
    W_mask = torch.zeros_like(W_metric) == 1
    if prune_n != 0:
        # structured n:m sparsity
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii : (ii + prune_m)].float()
                W_mask.scatter_(
                    1,
                    ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                    True,
                )
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity)]
        W_mask.scatter_(1, indices, True)

    W[W_mask] = 0.0  # set weights to zero

    if isinstance(module, transformers.Conv1D):
        W = W.t()

    W = W.reshape(final_shape).to(final_dtype)

    return W
