import math

import torch
import transformers
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import calculate_range
from compressed_tensors.utils import (
    get_execution_device,
    getattr_chain,
    update_offload_parameter,
)

GPTQ_PRECISION = torch.float32
MIN_BATCHED_CHOLESKY_SIZE = 16


class FusedQuantType:
    """Numeric quantizer identifiers consumed by the fused GPTQ Triton kernel."""

    INT = 0
    FP4_E2M1 = 1
    FP8_E4M3 = 2


def make_empty_hessian(
    module: torch.nn.Module, device: torch.device | None = None
) -> torch.Tensor:
    """Allocate the FP32 square Hessian accumulator for a module's input width."""
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros((num_columns, num_columns), device=device, dtype=GPTQ_PRECISION)


def accumulate_hessian(
    inp: torch.Tensor,
    module: torch.nn.Module,
    hessian: torch.Tensor,
    num_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate one module-input batch into its GPTQ Hessian statistics.

    The Hessian and sample counter are updated in place and returned for the
    hook caller to retain. Linear, Conv1D, and Conv2d inputs are reshaped into
    the common ``[input_features, observations]`` representation first.
    """
    inp = inp.to(device=hessian.device)
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
            inp = unfold(inp).permute([1, 0, 2]).flatten(1)

    num_samples += num_added
    inp = math.sqrt(2) * inp.to(dtype=GPTQ_PRECISION)
    hessian += inp.matmul(inp.t())
    return hessian, num_samples


def prepare_batch(
    batch: list[torch.nn.Module],
    batch_qparams: list[dict[str, torch.Tensor]],
    hessians_by_module: dict[torch.nn.Module, torch.Tensor],
    num_samples_by_module: dict[torch.nn.Module, torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Build disposable stacked GPTQ inputs and consume per-module statistics.

    Removes each module's accumulated Hessian and sample count from the passed
    dictionaries, normalizes its Hessian, and stacks it with weights and
    observer qparams for a single call to ``quantize_weight``.
    """
    hessian_list = []
    for module in batch:
        hessian = hessians_by_module.pop(module)
        num_samples = num_samples_by_module.pop(module).to(device=hessian.device)
        hessian_list.append(hessian / num_samples)
    hessians = torch.stack(hessian_list)
    del hessian_list

    weights = torch.empty(
        (len(batch), *batch[0].weight.shape),
        device=batch[0].weight.device,
        dtype=GPTQ_PRECISION,
    )
    torch.stack([module.weight for module in batch], out=weights)
    scales = torch.stack([qparam["scale"] for qparam in batch_qparams])
    zero_points = torch.stack([qparam["zero_point"] for qparam in batch_qparams])
    global_scales = (
        torch.stack([qparam["global_scale"].reshape(-1)[0] for qparam in batch_qparams])
        if batch_qparams[0]["global_scale"] is not None
        else None
    )
    return weights, hessians, scales, zero_points, global_scales


def update_batch_qparams(
    modules: list[torch.nn.Module],
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    global_scales: torch.Tensor | None,
    quant_args: QuantizationArgs,
) -> None:
    """Write a batch's quantized weights and qparams back to its modules.

    ``update_offload_parameter`` preserves offload-cache semantics while each
    stacked result is cast to the module's storage dtype.
    """
    for index, module in enumerate(modules):
        q_param_dict = {
            "weight": quantized[index].to(dtype=module.weight.dtype),
            "weight_scale": scales[index].to(dtype=module.weight.dtype),
            "weight_zero_point": zero_points[index].to(dtype=quant_args.zp_dtype),
        }
        if global_scales is not None:
            q_param_dict["weight_global_scale"] = global_scales[index].to(
                dtype=module.weight.dtype
            )
        for attr, val in q_param_dict.items():
            update_offload_parameter(module, attr, val)


def assign_batches(
    module_list: list[torch.nn.Module],
    batched_quantization: str | int | None,
    block_size: int,
) -> list[list[torch.nn.Module]]:
    """Partition modules into compatible GPTQ execution batches.

    ``None`` produces singleton batches; an integer caps every compatible
    batch; and ``"auto"`` derives a cap from the available CUDA memory.
    """
    if batched_quantization is None:
        return [[module] for module in module_list]

    batches: list[list[torch.nn.Module]] = []
    pending: dict[tuple, list[torch.nn.Module]] = {}
    for module in module_list:
        key = batch_key(module)
        if key is None:
            batches.append([module])
        else:
            pending.setdefault(key, []).append(module)

    for members in pending.values():
        max_batch = max_batch_size(members[0], batched_quantization, block_size)
        for start in range(0, len(members), max_batch):
            batches.append(members[start : start + max_batch])
    return batches


def batch_key(module: torch.nn.Module) -> tuple | None:
    """Return the shape, device, dtype, and qparam key needed to share a batch.

    Return ``None`` for modules without a compatible two-dimensional weight or
    quantization configuration, forcing them into a singleton batch.
    """
    weight = getattr(module, "weight", None)
    if weight is None or weight.dim() != 2:
        return None
    quant_args = getattr_chain(module, "quantization_scheme.weights", None)
    if quant_args is None:
        return None
    try:
        args_repr = quant_args.model_dump_json()
    except Exception:
        return None
    return (
        tuple(weight.shape),
        str(weight.dtype),
        str(get_execution_device(module)),
        args_repr,
    )


def max_batch_size(
    module: torch.nn.Module, batched_quantization: str | int, block_size: int
) -> int:
    """Return the safe batch cap for one representative compatible module.

    An explicit integer cap bypasses memory estimation. ``"auto"`` reserves
    75% of currently free CUDA memory for the largest estimated GPTQ phase.
    """
    if isinstance(batched_quantization, int):
        return batched_quantization

    out_features, in_features = module.weight.shape
    device = get_execution_device(module)
    weight_size = out_features * in_features
    hessian_size = in_features * in_features
    block_matrix_size = out_features * block_size
    quantization_peak = (
        hessian_size  # stacked Hessian, reused in place as its inverse
        + 3 * weight_size  # module weight, stacked working weight, and w_err
        + 4 * block_matrix_size  # W1, Q1, Err1, and losses1
    )
    actorder_peak = (
        3 * hessian_size  # original Hessian and both gather outputs
        + 2 * weight_size  # module weight and stacked working weight
    )
    per_module_bytes = max(quantization_peak, actorder_peak) * 4

    if torch.device(device).type != "cuda" or not torch.accelerator.is_available():
        return 1
    free_bytes, _ = torch.get_device_module().mem_get_info(device)
    return max(1, int(free_bytes * 0.75) // per_module_bytes)


def apply_activation_ordering(
    weights: torch.Tensor,
    hessians: torch.Tensor,
    actorder: ActivationOrdering | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Reorder weight columns and both Hessian axes by descending activation.

    The supplied working tensors are overwritten in place. The returned
    permutation is used to restore the quantized weights' original order.
    """
    if not actorder:
        return weights, hessians, None
    if actorder not in (ActivationOrdering.WEIGHT, ActivationOrdering.STATIC):
        raise ValueError(
            f"Invalid activation ordering {actorder}. Only 'weight' and 'static'"
            " are supported for GPTQ."
        )

    num_rows, num_columns = weights.shape[-2:]
    perm = torch.argsort(
        torch.diagonal(hessians, dim1=-2, dim2=-1), dim=-1, descending=True
    )
    hessian_perm = perm.to(device=hessians.device)
    weight_perm = perm.to(device=weights.device)
    permuted_hessians = torch.gather(
        hessians,
        -1,
        hessian_perm.unsqueeze(-2).expand(-1, num_columns, -1),
    )
    permuted_hessians = torch.gather(
        permuted_hessians,
        -2,
        hessian_perm.unsqueeze(-1).expand(-1, -1, num_columns),
    )
    hessians.copy_(permuted_hessians)
    del permuted_hessians

    permuted_weights = torch.gather(
        weights,
        -1,
        weight_perm.unsqueeze(-2).expand(-1, num_rows, -1),
    )
    weights.copy_(permuted_weights)
    return weights, hessians, weight_perm


def factorize_hessian(
    weights: torch.Tensor,
    hessians: torch.Tensor,
    percdamp: float,
    used_rtn_fallback: torch.Tensor,
) -> torch.Tensor:
    """Dampen and factorize Hessians in place into GPTQ update factors.

    Dead columns are zeroed in ``weights``. Non-positive-definite Hessians are
    replaced with identity factors and marked for RTN fallback. Small batches
    use per-item linear algebra because it is faster than batched CUDA calls.
    """
    batch_size, _, num_columns = weights.shape
    diag = torch.diagonal(hessians, dim1=-2, dim2=-1)
    dead = diag == 0
    if dead.any():
        diag.masked_fill_(dead, 1.0)
        weights.masked_fill_(dead.unsqueeze(1), 0)

    # For small batches its faster to do it per-item than to use the batched CUDA ops.
    if batch_size < MIN_BATCHED_CHOLESKY_SIZE:
        info = torch.empty((), dtype=torch.int32, device=hessians.device)
        identity = None
        for index in range(batch_size):
            damp = percdamp * torch.mean(torch.diag(hessians[index]))
            torch.diagonal(hessians[index]).add_(damp)
            torch.linalg.cholesky_ex(
                hessians[index], check_errors=False, out=(hessians[index], info)
            )
            if info.item() == 0:
                torch.cholesky_inverse(hessians[index], out=hessians[index])
                torch.linalg.cholesky(hessians[index], upper=True, out=hessians[index])
            else:
                if identity is None:
                    identity = torch.eye(
                        num_columns, dtype=hessians.dtype, device=hessians.device
                    )
                hessians[index].copy_(identity)
                used_rtn_fallback[index] = True
        return hessians

    damp = percdamp * diag.mean(dim=-1)
    diag.add_(damp.unsqueeze(-1))
    info = torch.empty(batch_size, dtype=torch.int32, device=hessians.device)
    torch.linalg.cholesky_ex(hessians, check_errors=False, out=(hessians, info))
    bad = info.nonzero(as_tuple=False).flatten()
    if bad.numel():
        hessians.index_copy_(
            0,
            bad,
            torch.eye(num_columns, dtype=hessians.dtype, device=hessians.device).expand(
                bad.numel(), -1, -1
            ),
        )
        used_rtn_fallback[bad] = True
    torch.cholesky_inverse(hessians, out=hessians)
    torch.linalg.cholesky(hessians, upper=True, out=hessians)
    return hessians


def column_scale_window(
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    quant_args: QuantizationArgs,
    num_rows: int,
    i1: int,
    i2: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Expand qparams into effective per-row, per-column values for a block.

    Supports tensor, channel, group, tensor-group, and block qparam layouts;
    folds in the optional global scale; and returns tensors consumable by both
    the eager and Triton GPTQ block-update implementations.
    """
    strategy = quant_args.strategy
    block_width = i2 - i1
    has_zp = zero_point is not None and not quant_args.symmetric

    if strategy == QuantizationStrategy.TENSOR:
        eff = scale.reshape(-1, 1, 1)
        zp = zero_point.reshape(-1, 1, 1) if has_zp else None
    elif strategy == QuantizationStrategy.CHANNEL:
        eff = scale[..., :, 0:1]
        zp = zero_point[..., :, 0:1] if has_zp else None
    elif strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        idx = g_idx[..., i1:i2].long()
        eff = torch.gather(
            scale, -1, idx.unsqueeze(-2).expand(*scale.shape[:-1], block_width)
        )
        zp = (
            torch.gather(
                zero_point,
                -1,
                idx.unsqueeze(-2).expand(*zero_point.shape[:-1], block_width),
            )
            if has_zp
            else None
        )
    elif strategy == QuantizationStrategy.BLOCK:
        block_height, _ = quant_args.block_structure
        row_idx = torch.arange(num_rows, device=scale.device) // block_height
        col_idx = g_idx[..., i1:i2].long().unsqueeze(-2)
        eff = torch.gather(scale, -1, col_idx.expand(*scale.shape[:-1], block_width))
        row_idx = row_idx.reshape((1,) * (eff.ndim - 2) + (num_rows, 1))
        eff = torch.gather(
            eff,
            -2,
            row_idx.expand(*eff.shape[:-2], num_rows, block_width),
        )
        if has_zp:
            zp = torch.gather(
                zero_point,
                -1,
                col_idx.expand(*zero_point.shape[:-1], block_width),
            )
            zp = torch.gather(
                zp,
                -2,
                row_idx.expand(*zp.shape[:-2], num_rows, block_width),
            )
        else:
            zp = None
    else:
        raise ValueError(f"Unsupported strategy for column scale window: {strategy}")

    if global_scale is not None:
        global_scale = global_scale.reshape(
            *global_scale.shape, *([1] * (eff.ndim - global_scale.ndim))
        )
        eff = eff / global_scale
    eff = eff.expand(*eff.shape[:-2], num_rows, block_width).contiguous()
    if zp is None:
        return eff, None
    zp = zp.to(GPTQ_PRECISION)
    zp = zp.expand(*zp.shape[:-2], num_rows, block_width).contiguous()
    return eff, zp


def get_triton_gptq_config(
    quant_args: QuantizationArgs,
) -> tuple[int, float, float] | None:
    """Map supported qargs to fused-kernel type and numeric code range.

    Return ``None`` when the registered Triton block-update backend cannot
    represent the requested quantization scheme.
    """
    if quant_args.strategy not in (
        QuantizationStrategy.TENSOR,
        QuantizationStrategy.CHANNEL,
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
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
