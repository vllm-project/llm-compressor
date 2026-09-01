import math
import os
from copy import copy

import torch
import transformers
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    fake_quantize,
)
from compressed_tensors.quantization.utils import calculate_range
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import triton_req
from loguru import logger

from llmcompressor.modifiers.gptq.gptq_triton import (
    FusedQuantType,
    fused_gptq_block_update,
)
from llmcompressor.modifiers.quantization.calibration import observe

GPTQ_PRECISION = torch.float32

__all__ = [
    "make_empty_hessian",
    "accumulate_hessian",
    "quantize_weight",
    "quantize_weight_batched",
    "is_batched_quantizable",
]


def is_batched_quantizable(quant_args: QuantizationArgs) -> bool:
    """
    Whether a weight quantization scheme is supported by the fused batched
    path. Unsupported cases use the normal single-module GPTQ path.
    """
    if quant_args.strategy not in (
        QuantizationStrategy.TENSOR,
        QuantizationStrategy.CHANNEL,
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
        return False
    return quant_args.type in (QuantizationType.INT, QuantizationType.FLOAT)


def make_empty_hessian(
    module: torch.nn.Module, device: torch.device | None = None
) -> torch.Tensor:
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros((num_columns, num_columns), device=device, dtype=GPTQ_PRECISION)


def accumulate_hessian(
    inp: torch.Tensor,
    module: torch.nn.Module,
    H: torch.Tensor | None,
    num_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp = inp.to(device=H.device)
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    elif len(inp.shape) > 3:
        inp = inp.reshape(inp.shape[0], -1, inp.shape[-1])

    num_added = inp.shape[0]

    match module:
        case torch.nn.Linear() | transformers.Conv1D():
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        case torch.nn.Conv2d():
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

    num_samples += num_added

    inp = inp.to(dtype=GPTQ_PRECISION)
    inp = math.sqrt(2) * inp
    H += inp.matmul(inp.t())

    return H, num_samples


def quantize_weight(
    module: torch.nn.Module,
    quant_args: QuantizationArgs,
    hessian: torch.Tensor,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> tuple[float, dict[str, torch.Tensor], bool]:
    """
    Quantize a module weight according to the GPTQ algorithm

    :param module: module with weight being quantized
    :param quant_args: quantization arguments used to find quantization parameters
    :param hessian: preaccumulated hessian for quantization
    :param blocksize: chunk size of quantization updates
    :param percdamp: dampening factor on hessian diagonal
    :return: loss, q_param_dict (with keys: weight, weight_scale, weight_zero_point,
        and optionally weight_global_scale), used_rtn_fallback (True if hessian
        inversion failed and the module was quantized with round-to-nearest)
    """
    strategy = quant_args.strategy
    actorder = quant_args.actorder
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()
    H = hessian

    observer = module.weight_observer
    W = W.to(dtype=GPTQ_PRECISION)
    num_rows = W.shape[0]
    num_columns = W.shape[1]

    # handle activation ordering
    if actorder:
        if actorder not in (ActivationOrdering.WEIGHT, ActivationOrdering.STATIC):
            raise ValueError(
                f"Invalid activation ordering {actorder}. Only 'weight' and 'static'"
                "are supported for GPTQ."
            )
        W, H, perm = _apply_activation_ordering(W, H)

    # handle g_idx
    if strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
        # mapping from column index to group index
        divisor = (
            quant_args.group_size
            if strategy != QuantizationStrategy.BLOCK
            else quant_args.block_structure[1]
        )
        g_idx = torch.arange(num_columns, device=W.device, dtype=torch.int) // divisor

        if actorder == ActivationOrdering.WEIGHT:
            g_idx = g_idx[perm]

    qparams = observer.get_qparams()
    scale, zero_point, global_scale = (
        qparams["scale"],
        qparams["zero_point"],
        qparams["global_scale"],
    )

    losses = torch.zeros(num_rows, device=module.weight.device)

    # mask dead hessian values
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # compute inverse hessian in place to save memory
    used_rtn_fallback = False
    try:
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
    except torch._C._LinAlgError:
        logger.warning(
            "Failed to invert hessian due to numerical instability. Consider "
            "increasing GPTQModifier.dampening_frac, increasing the number "
            "of calibration samples, or shuffling the calibration dataset. "
            "Falling back to round-to-nearest for this module."
        )
        used_rtn_fallback = True
        Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)

    # See section 3.4 of https://arxiv.org/abs/2203.07259
    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        gptq_block_update(
            W1,
            Hinv1,
            Q1,
            Err1,
            losses1,
            scale=scale,
            zero_point=zero_point,
            global_scale=global_scale,
            g_idx=(
                g_idx
                if strategy in _GROUP_STRATEGIES
                or strategy == QuantizationStrategy.BLOCK
                else None
            ),
            quant_args=quant_args,
            i1=i1,
        )

        # propagate block error
        W[:, i1:i2] = Q1
        losses += torch.sum(losses1, 1) / 2

        w_err = Err1.matmul(Hinv[i1:i2, i2:])
        W[:, i2:] -= w_err

    if actorder:
        # restore original permutation
        invperm = torch.argsort(perm)
        W = W[:, invperm]

    W = W.reshape(final_shape).to(final_dtype)

    loss = torch.sum(losses).item()
    q_param_dict = {
        "weight": W,
        "weight_scale": scale.to(dtype=final_dtype),
        "weight_zero_point": zero_point.to(dtype=quant_args.zp_dtype),
    }
    if global_scale:
        q_param_dict["weight_global_scale"] = global_scale.to(dtype=final_dtype)
    return (loss, q_param_dict, used_rtn_fallback)


def _apply_activation_ordering(
    W: torch.Tensor, H: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Permute weight and hessian in order of greatest output activations

    :param W: weight to permute
    :param H: hessian used to determine activation ordering
    :return: permuted weight, permuted hessian, permutation map
    """
    perm = torch.argsort(torch.diag(H), descending=True)
    return W[:, perm], H[perm][:, perm], perm


_GROUP_STRATEGIES = (
    QuantizationStrategy.GROUP,
    QuantizationStrategy.TENSOR_GROUP,
)


def _fused_kernel_params(
    quant_args: QuantizationArgs,
) -> tuple[int, float, float] | None:
    """
    Resolve fused-kernel quantization parameters, or None if the scheme is not
    supported by the fused kernel.
    """
    if quant_args.strategy in (
        QuantizationStrategy.TENSOR,
        QuantizationStrategy.CHANNEL,
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
        pass
    else:
        return None

    if quant_args.type == QuantizationType.INT:
        quant_type = FusedQuantType.INT
    elif quant_args.type == QuantizationType.FLOAT and quant_args.num_bits == 4:
        quant_type = FusedQuantType.FP4_E2M1
    elif quant_args.type == QuantizationType.FLOAT and quant_args.num_bits == 8:
        quant_type = FusedQuantType.FP8_E4M3
    else:
        return None

    q_min, q_max = calculate_range(quant_args, "cpu")
    return quant_type, float(q_min), float(q_max)


def _column_scale_window(
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    quant_args: QuantizationArgs,
    num_rows: int,
    i1: int,
    i2: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Expand quantization parameters into effective per-column values over the
    column window [i1, i2), with the global scale folded in.

    Supports arbitrary leading batch dimensions: `scale` is [..., num_rows, G]
    and `g_idx` (group strategies) is [... or absent, num_columns].

    :return: (eff_scale [..., num_rows, count], zero_point or None)
    """
    strategy = quant_args.strategy
    count = i2 - i1

    has_zp = zero_point is not None and not quant_args.symmetric

    if strategy == QuantizationStrategy.TENSOR:
        # per-tensor scales may be shaped [], [1], or [1, 1] per module, so
        # a stacked batch can be [B], [B, 1], or [B, 1, 1]; normalize to
        # [..., 1, 1]
        if scale.ndim >= 2 and scale.shape[-2:] == (1, 1):
            eff = scale
        else:
            eff = scale.reshape(-1)[..., None, None]
        if has_zp:
            if zero_point.ndim >= 2 and zero_point.shape[-2:] == (1, 1):
                zp = zero_point
            else:
                zp = zero_point.reshape(-1)[..., None, None]
        else:
            zp = None
    elif strategy == QuantizationStrategy.CHANNEL:
        eff = scale[..., :, 0:1]
        zp = zero_point[..., :, 0:1] if has_zp else None
    elif strategy in _GROUP_STRATEGIES:
        idx = g_idx[..., i1:i2].long()
        eff = torch.gather(
            scale, -1, idx.unsqueeze(-2).expand(*scale.shape[:-1], count)
        )
        zp = (
            torch.gather(
                zero_point,
                -1,
                idx.unsqueeze(-2).expand(*zero_point.shape[:-1], count),
            )
            if has_zp
            else None
        )
    elif strategy == QuantizationStrategy.BLOCK:
        block_height, _ = quant_args.block_structure
        row_idx = torch.arange(num_rows, device=scale.device) // block_height
        row_idx = row_idx.reshape((1,) * (scale.ndim - 2) + (num_rows, 1))
        col_idx = g_idx[..., i1:i2].long().unsqueeze(-2)
        eff = scale[..., row_idx, col_idx]
        zp = zero_point[..., row_idx, col_idx] if has_zp else None
    else:
        raise ValueError(f"Unsupported strategy for column scale window: {strategy}")

    eff = eff.to(GPTQ_PRECISION)
    if global_scale is not None:
        gs = global_scale.to(GPTQ_PRECISION)
        gs = gs.reshape(*gs.shape, *([1] * (eff.ndim - gs.ndim)))
        eff = eff / gs
    eff = eff.expand(*eff.shape[:-2], num_rows, count).contiguous()

    if zp is None:
        # symmetric zero points are exactly zero; adding them is a no-op
        return eff, None

    zp = zp.to(GPTQ_PRECISION)
    zp = zp.expand(*zp.shape[:-2], num_rows, count).contiguous()
    return eff, zp


def _gptq_block_update_triton_req(
    W1: torch.Tensor,
    Hinv1: torch.Tensor,
    Q1: torch.Tensor,
    Err1: torch.Tensor,
    losses1: torch.Tensor,
    *,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    quant_args: QuantizationArgs,
    i1: int,
) -> bool:
    count = W1.shape[-1]
    return (
        triton_req(W1)
        and os.environ.get("LLMCOMPRESSOR_DISABLE_GPTQ_TRITON", "0") != "1"
        and _fused_kernel_params(quant_args) is not None
        and 0 < count <= 256
        and not count & (count - 1)
    )


@ImplBackend.register("gptq_block_update", _gptq_block_update_triton_req, 0)
def _gptq_block_update_triton(
    W1: torch.Tensor,
    Hinv1: torch.Tensor,
    Q1: torch.Tensor,
    Err1: torch.Tensor,
    losses1: torch.Tensor,
    *,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    quant_args: QuantizationArgs,
    i1: int,
) -> None:
    """Run one GPTQ block with the registered Triton backend."""
    params = _fused_kernel_params(quant_args)
    if params is None:
        raise ValueError(f"Unsupported Triton GPTQ scheme: {quant_args}")

    count = W1.shape[-1]
    if count > 256 or count & (count - 1):
        raise ValueError("Triton GPTQ block width must be a power of two <= 256")

    quant_type, q_min, q_max = params
    eff, zp = _column_scale_window(
        scale,
        zero_point,
        global_scale,
        g_idx,
        quant_args,
        num_rows=W1.shape[-2],
        i1=i1,
        i2=i1 + count,
    )
    fused_gptq_block_update(
        W1.unsqueeze(-3) if W1.dim() == 2 else W1,
        Hinv1.unsqueeze(-3) if Hinv1.dim() == 2 else Hinv1,
        eff.unsqueeze(-3) if eff.dim() == 2 else eff,
        zp if zp is None or zp.dim() == 3 else zp.unsqueeze(-3),
        Q1.unsqueeze(-3) if Q1.dim() == 2 else Q1,
        Err1.unsqueeze(-3) if Err1.dim() == 2 else Err1,
        q_min,
        q_max,
        quant_type,
    )
    losses1.copy_(Err1.square())


@ImplBackend.entrypoint("gptq_block_update")
def gptq_block_update(
    W1: torch.Tensor,
    Hinv1: torch.Tensor,
    Q1: torch.Tensor,
    Err1: torch.Tensor,
    losses1: torch.Tensor,
    *,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    quant_args: QuantizationArgs,
    i1: int,
) -> None:
    """Run one GPTQ block with the eager Torch implementation."""
    if W1.dim() != 2:
        raise ValueError("The eager GPTQ block backend only supports one matrix")

    strategy = quant_args.strategy
    count = W1.shape[-1]
    for i in range(count):
        w = W1[:, i]
        d = Hinv1[i, i]

        if strategy == QuantizationStrategy.TENSOR:
            q = fake_quantize(
                w, scale, zero_point, quant_args, global_scale=global_scale
            )
        elif strategy == QuantizationStrategy.CHANNEL:
            q = fake_quantize(
                w,
                scale[:, 0],
                zero_point[:, 0],
                quant_args,
                global_scale=global_scale,
            )
        elif strategy in _GROUP_STRATEGIES:
            group_index = g_idx[i1 + i]
            altered_qargs = copy(quant_args)
            altered_qargs.strategy = QuantizationStrategy.CHANNEL
            q = fake_quantize(
                w,
                scale[:, group_index],
                zero_point[:, group_index],
                altered_qargs,
                global_scale=global_scale,
            )
        elif strategy == QuantizationStrategy.BLOCK:
            block_column_idx = g_idx[i1 + i]
            q = fake_quantize(
                w.unsqueeze(1),
                scale[:, block_column_idx : block_column_idx + 1],
                zero_point[:, block_column_idx : block_column_idx + 1],
                quant_args,
                global_scale=global_scale,
            ).squeeze(1)
        else:
            raise ValueError(
                f"Quantization strategy is not supported for GPTQ: {strategy}"
            )

        Q1[:, i] = q
        losses1[:, i] = (w - q) ** 2 / d**2
        err = (w - q) / d
        W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
        Err1[:, i] = err


def _quantize_weights_individually(
    modules: list[torch.nn.Module],
    quant_args: QuantizationArgs,
    hessians: list[torch.Tensor],
    blocksize: int,
    percdamp: float,
    refresh_observers: bool = False,
) -> list[tuple[float, dict[str, torch.Tensor], bool]]:
    if refresh_observers:
        for module in modules:
            observe(module, "weight")
    return [
        quantize_weight(module, quant_args, hessian.clone(), blocksize, percdamp)
        for module, hessian in zip(modules, hessians)
    ]


def quantize_weight_batched(
    modules: list[torch.nn.Module],
    quant_args: QuantizationArgs,
    hessians: list[torch.Tensor],
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> list[tuple[float, dict[str, torch.Tensor], bool]]:
    """
    Batched GPTQ over multiple same-shape modules which share one
    quantization scheme (e.g. linearized MoE experts). Weights are stacked to
    [B, out, in] and Hessians to [B, in, in], and the Cholesky solves, column
    updates, and error propagation run batched in lockstep.

    All modules must have identically shaped weights and quantization args;
    the caller is responsible for grouping. Batched execution uses the fused
    CUDA path; unsupported or unavailable cases use `quantize_weight` per
    module.

    :param modules: modules whose weights are quantized; each must have a
        populated `weight_observer`
    :param quant_args: shared quantization arguments
    :param hessians: preaccumulated Hessians, one per module
    :param blocksize: chunk size of quantization updates
    :param percdamp: dampening factor on hessian diagonal
    :return: list of (loss, q_param_dict, used_rtn_fallback), one per module
    """
    if len(modules) == 0:
        return []

    # Batched execution is specifically the fused CUDA path.  Keep the
    # fallback as the normal single-module GPTQ implementation rather than a
    # second batched eager implementation.
    if modules[0].weight.device.type != "cuda":
        return _quantize_weights_individually(
            modules, quant_args, hessians, blocksize, percdamp
        )

    strategy = quant_args.strategy
    actorder = quant_args.actorder
    batch_size = len(modules)
    final_shape = modules[0].weight.shape
    final_dtype = modules[0].weight.dtype
    num_rows, num_columns = final_shape
    device = modules[0].weight.device

    original_hessians = [h.clone() for h in hessians]
    used_rtn_fallback = [False] * batch_size
    W = torch.stack([module.weight for module in modules]).to(dtype=GPTQ_PRECISION)
    H = torch.stack([h.to(device=device) for h in hessians]).to(dtype=GPTQ_PRECISION)

    # handle activation ordering (per batch element)
    perm = None
    if actorder:
        if actorder not in (ActivationOrdering.WEIGHT, ActivationOrdering.STATIC):
            raise ValueError(
                f"Invalid activation ordering {actorder}. Only 'weight' and 'static'"
                "are supported for GPTQ."
            )
        perm = torch.argsort(
            torch.diagonal(H, dim1=-2, dim2=-1), dim=-1, descending=True
        )
        W = torch.gather(W, -1, perm.unsqueeze(-2).expand(-1, num_rows, -1))
        H = torch.gather(H, -1, perm.unsqueeze(-2).expand(-1, num_columns, -1))
        H = torch.gather(H, -2, perm.unsqueeze(-1).expand(-1, -1, num_columns))

    # mapping from column index to group index
    g_idx = None
    if strategy in _GROUP_STRATEGIES or strategy == QuantizationStrategy.BLOCK:
        g_idx = torch.arange(num_columns, device=device, dtype=torch.int)
        divisor = (
            quant_args.group_size
            if strategy in _GROUP_STRATEGIES
            else quant_args.block_structure[1]
        )
        g_idx = (g_idx // divisor).unsqueeze(0).expand(batch_size, -1)
        if actorder == ActivationOrdering.WEIGHT:
            g_idx = torch.gather(g_idx, 1, perm)

    losses = torch.zeros(batch_size, num_rows, device=device)

    # mask dead hessian values
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    dead = diag == 0
    if dead.any():
        torch.diagonal(H, dim1=-2, dim2=-1).masked_fill_(dead, 1.0)
        W.masked_fill_(dead.unsqueeze(1), 0)

    # compute inverse hessians in place to save memory
    damp = percdamp * torch.diagonal(H, dim1=-2, dim2=-1).mean(dim=-1)
    torch.diagonal(H, dim1=-2, dim2=-1).add_(damp.unsqueeze(-1))
    info = torch.empty(batch_size, dtype=torch.int32, device=device)
    torch.linalg.cholesky_ex(H, check_errors=False, out=(H, info))
    bad = info.nonzero(as_tuple=False).flatten().tolist()
    if bad:
        return _quantize_weights_individually(
            modules, quant_args, original_hessians, blocksize, percdamp
        )
    torch.cholesky_inverse(H, out=H)
    torch.linalg.cholesky(H, upper=True, out=H)
    Hinv = H

    qparams = [module.weight_observer.get_qparams() for module in modules]
    # observers may live on a different device (e.g. cpu) when offloading
    scale = torch.stack([qp["scale"] for qp in qparams]).to(device=device)
    zero_point = torch.stack([qp["zero_point"] for qp in qparams]).to(device=device)
    global_scale = None
    if qparams[0]["global_scale"] is not None:
        global_scale = torch.stack(
            [qp["global_scale"].reshape(-1)[0] for qp in qparams]
        ).to(device=device)

    # See section 3.4 of https://arxiv.org/abs/2203.07259
    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)

        W1 = W[:, :, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[:, i1:i2, i1:i2]

        block_args = (W1, Hinv1, Q1, Err1, losses1)
        block_kwargs = {
            "scale": scale,
            "zero_point": zero_point,
            "global_scale": global_scale,
            "g_idx": g_idx,
            "quant_args": quant_args,
            "i1": i1,
        }
        if not _gptq_block_update_triton_req(*block_args, **block_kwargs):
            return _quantize_weights_individually(
                modules,
                quant_args,
                original_hessians,
                blocksize,
                percdamp,
                refresh_observers=True,
            )
        try:
            ImplBackend.call(
                "_gptq_block_update_triton", *block_args, **block_kwargs
            )
        except Exception:
            return _quantize_weights_individually(
                modules,
                quant_args,
                original_hessians,
                blocksize,
                percdamp,
                refresh_observers=True,
            )

        W[:, :, i1:i2] = Q1
        losses += losses1.sum(dim=-1) / 2
        if i2 < num_columns:
            W[:, :, i2:] -= torch.bmm(Err1, Hinv[:, i1:i2, i2:])

    if actorder:
        # restore original permutation
        invperm = torch.argsort(perm, dim=-1)
        W = torch.gather(W, -1, invperm.unsqueeze(-2).expand(-1, num_rows, -1))

    results = []
    for k, module in enumerate(modules):
        q_param_dict = {
            "weight": W[k].reshape(final_shape).to(final_dtype),
            "weight_scale": scale[k].to(dtype=final_dtype),
            "weight_zero_point": zero_point[k].to(dtype=quant_args.zp_dtype),
        }
        if global_scale is not None:
            q_param_dict["weight_global_scale"] = global_scale[k].to(dtype=final_dtype)
        results.append((losses[k].sum().item(), q_param_dict, used_rtn_fallback[k]))
    return results
