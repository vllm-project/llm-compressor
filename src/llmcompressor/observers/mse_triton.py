"""Triton grid search for the MSE observer.

The eager search fake-quantizes the full tensor once per shrink step. This
kernel keeps each group in registers, normalizes it by the initial qparam
scale, and walks the format's codebook incrementally as the range shrinks.

The zero point is loaded once per group: shrinking min and max by ``p``
leaves ``min / scale`` unchanged, so the zero point is invariant in exact
arithmetic; eager's per-step recompute can differ by a code in bf16. The
kernel adds it to ``x / (scale * p)`` and searches one unshifted table per
format.
"""

from typing import Tuple

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationType
from compressed_tensors.quantization.utils import calculate_qparams

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the fallback test
    _TRITON_AVAILABLE = False

__all__ = ["can_use_triton", "grid_search_triton"]

# Float dtypes the kernel has been exercised on. The search itself runs in
# fp32 for all of them; this is about what amin/amax and calculate_qparams
# were checked to produce, not about the arithmetic inside.
_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_SUPPORTED_SCALE_DTYPES = (None, torch.float8_e4m3fn)

# Bit widths whose codebook this has actually been checked against. Nothing
# stops the int branch from building 65536 codes for int16, but
# neighbors_for_config is quadratic in the cutoff count and the kernel would
# binary search a table nobody has validated.
_INT_BITS = (4, 8)

# Largest group the kernel has been exercised at. BLOCK_G is the next power
# of two above this, so a CHANNEL layout on a wide layer, or a 128x128
# block, would otherwise ask one program to hold 4096 or 16384 elements.
# Raising it is a measurement, not an edit.
_MAX_GROUP_SIZE = 256


def _codebook(args: QuantizationArgs) -> torch.Tensor:
    """The values an element can be rounded to, once divided by the scale.

    :return: sorted representable values, or None if the format is not one
        this kernel supports
    """
    bits = args.num_bits
    if args.type == QuantizationType.INT and bits in _INT_BITS:
        q_min = -(2 ** (bits - 1))
        q_max = 2 ** (bits - 1) - 1
        return torch.arange(q_min, q_max + 1, dtype=torch.float32)
    if args.type == QuantizationType.FLOAT and bits == 4:
        # the float4_e2m1 values, written out like PR #2948's codebook
        # builder rather than imported: compressed_tensors moved its copy of
        # this table between nightlies, and an import from its internals
        # breaks the whole observer at import time when it moves again
        return torch.tensor(
            [
                -6.0,
                -4.0,
                -3.0,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
            ],
            dtype=torch.float32,
        )
    if args.type == QuantizationType.FLOAT and bits == 8:
        codes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        return codes[torch.isfinite(codes)].unique().sort().values
    return None


def neighbors_for_config(args: QuantizationArgs, grid: float, maxshrink: float) -> int:
    """How many neighbors a value can move in one step of the search.

    In normalized units step ``k`` compares ``x / (scale * p_k) + zp``
    against the fixed cutoffs, and ``p`` decreases, so every value drifts
    away from its fixed point - the position where ``x / scale`` is zero,
    which is ``zp``. A value at ``cutoff[i]`` drifts by
    ``(cutoff[i] - zp) * (1/s - 1)`` per step, with ``s = p_{k+1} / p_k`` at
    its tightest on the last step, and can reach ``cutoff[i+k]`` only when
    that drift covers the gap between them. The answer is the smallest ``k``
    whose gap no drift covers.

    ``zp`` is data, but calculate_qparams clamps it to the code range, so
    the two ends of the table bound every position it can take; symmetric
    formats pin it to zero. That keeps the bound config-only, which matters
    because ``NUM_NEIGHBORS`` is a ``tl.constexpr``: deriving it from the
    data would compile a separate kernel per layer.
    """
    codes = _codebook(args)
    if codes is None:
        return 1
    cutoffs = ((codes[:-1] + codes[1:]) * 0.5).double()

    total_steps = int(maxshrink * grid)
    if grid - total_steps <= 0:
        # maxshrink >= 1 drives the last steps to p <= 0, where nothing
        # bounds the drift. Guard on the difference, not the ratio: at
        # grid - total_steps == -1 the ratio is a division by zero, and
        # below that both signs flip and it sneaks past a ratio check.
        return int(cutoffs.numel())
    ratio = (grid - total_steps) / (grid - total_steps + 1)
    drift = 1.0 / ratio - 1.0

    lo = 0.0 if args.symmetric else float(codes[0])
    hi = 0.0 if args.symmetric else float(codes[-1])

    limit = int(cutoffs.numel())
    for k in range(1, limit):
        gap = cutoffs[k:] - cutoffs[:-k]
        up = (cutoffs[:-k] - lo).clamp_min(0.0) * drift
        down = (hi - cutoffs[k:]).clamp_min(0.0) * drift
        if bool((up < gap).all() and (down < gap).all()):
            return k
    return limit


def can_use_triton(observed: torch.Tensor, args: QuantizationArgs) -> bool:
    """Whether the kernel searches the same candidates the eager search does.

    Deliberately narrow. Everything outside this keeps the existing eager or
    compiled path, which stays the reference:

    - ``observed`` must be 4d. TENSOR strategy flattens to 3d, and the
      launcher reads (obs, rows, groups, group_size).
    - int4, int8, float4 and float8 only. Wider integer widths have not been
      checked and make the codebook too large for this kernel.
    - plain or fp8 scale storage. MX exponent scales use a different path.
    - a single observation. The kernel gives one program per group and reads
      one row; eager reduces over the observation axis, so anything stacked
      there would be silently dropped.
    - the group has to fit a block the kernel has been run at.
    - CUDA proper, at compute capability 8.0 or above. torch.Tensor.is_cuda
      is also True under ROCm, the error path calls into libdevice, and
      Triton does not support older NVIDIA architectures.
    """
    return (
        _TRITON_AVAILABLE
        and observed.is_cuda
        and getattr(torch.version, "hip", None) is None
        and torch.get_device_module(observed.device).get_device_capability(
            observed.device
        )[0]
        >= 8
        and observed.ndim == 4
        and observed.shape[0] == 1
        and observed.shape[-1] <= _MAX_GROUP_SIZE
        and observed.dtype in _SUPPORTED_DTYPES
        and args.scale_dtype in _SUPPORTED_SCALE_DTYPES
        # fp8 scale storage exists for NVFP4-style fp4 schemes; no scheme
        # stores an int or fp8 codebook's scale in fp8
        and (
            args.scale_dtype is None
            or (args.type == QuantizationType.FLOAT and args.num_bits == 4)
        )
        # no scheme quantizes float asymmetrically: fp4-asym raises inside
        # calculate_qparams, and fp8-asym runs but nothing exercises it
        and (bool(args.symmetric) or args.type == QuantizationType.INT)
        and _codebook(args) is not None
    )


if _TRITON_AVAILABLE:

    @triton.jit
    def _grid_search_kernel(
        observed_ptr,
        scale_base_ptr,
        zero_point_ptr,
        codes_ptr,
        best_step_ptr,
        num_rows,
        num_groups,
        group_size,
        total_steps,
        num_codes,
        stride_obs_row,
        stride_obs_group,
        inv_grid,
        norm,
        patience,
        BLOCK_G: tl.constexpr,
        LOG_C: tl.constexpr,
        TOTAL_STEPS: tl.constexpr,
        NUM_NEIGHBORS: tl.constexpr,
    ):
        """One program per group; the group's values stay in registers.

        Patience is per group here, unlike eager's single counter that resets
        whenever any group in the tensor improves. Each program only sees its
        own group, so a global counter has no meaning; the per-group version
        stops strictly earlier, which is a behaviour change and the reason
        the accuracy sweep exists.
        """
        pid = tl.program_id(0)
        row = pid // num_groups
        group = pid % num_groups
        if row >= num_rows:
            return

        g_idx = tl.arange(0, BLOCK_G)
        g_mask = g_idx < group_size
        obs = tl.load(
            observed_ptr + row * stride_obs_row + group * stride_obs_group + g_idx,
            mask=g_mask,
            other=0.0,
        ).to(tl.float32)

        scale_base = tl.load(scale_base_ptr + row * num_groups + group).to(tl.float32)
        # min * p / (scale * p) is min / scale, so the zero point does not
        # move with p (exact arithmetic; bf16 recompute can differ)
        zero_point = tl.load(zero_point_ptr + row * num_groups + group).to(tl.float32)

        ig = inv_grid.to(tl.float32)
        norm_f = norm.to(tl.float32)

        shift_dir = tl.where(obs >= 0, 1, -1)
        best_err = tl.full([], float("inf"), dtype=tl.float32)
        best_s = 0
        bin_idx = tl.zeros([BLOCK_G], dtype=tl.int32)
        stall = 0

        for step in range(TOTAL_STEPS):
            if step < total_steps:
                if stall < patience:
                    p = (1.0 - step * ig).to(tl.float32)
                    scale = scale_base * p
                    sel = obs / scale + zero_point

                    if step == 0:
                        lo = tl.zeros([BLOCK_G], dtype=tl.int32)
                        hi = tl.full([BLOCK_G], num_codes - 1, dtype=tl.int32)
                        for _ in range(LOG_C):
                            mid = (lo + hi) >> 1
                            ge = sel >= tl.load(codes_ptr + mid)
                            lo = tl.where(ge, mid + 1, lo)
                            hi = tl.where(ge, hi, mid)
                        idx_left = tl.maximum(lo - 1, 0)
                        idx_right = tl.minimum(lo, num_codes - 1)
                        d_left = tl.abs(sel - tl.load(codes_ptr + idx_left))
                        d_right = tl.abs(sel - tl.load(codes_ptr + idx_right))
                        bin_idx = tl.where(d_left <= d_right, idx_left, idx_right)
                        best_d = tl.minimum(d_left, d_right)
                    else:
                        # accumulate into new_bin rather than bin_idx: writing
                        # back inside the loop would move the base each
                        # neighbor is measured from
                        new_bin = bin_idx
                        best_d = tl.abs(sel - tl.load(codes_ptr + bin_idx))
                        for k in tl.static_range(1, NUM_NEIGHBORS + 1):
                            idx_k = tl.maximum(
                                tl.minimum(bin_idx + shift_dir * k, num_codes - 1),
                                0,
                            )
                            d_k = tl.abs(sel - tl.load(codes_ptr + idx_k))
                            better = d_k < best_d
                            new_bin = tl.where(better, idx_k, new_bin)
                            best_d = tl.where(better, d_k, best_d)
                        bin_idx = new_bin

                    distance = best_d * scale
                    powed = tl.extra.cuda.libdevice.pow(distance, norm_f)
                    err = tl.sum(tl.where(g_mask, powed, 0.0), axis=0)

                    is_better = err < best_err
                    best_err = tl.where(is_better, err, best_err)
                    best_s = tl.where(is_better, step, best_s)
                    stall = tl.where(is_better, 0, stall + 1)

        tl.store(best_step_ptr + row * num_groups + group, best_s)


def _restore_range(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    best_step: torch.Tensor,
    grid: float,
    total_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the shrink factor selected by each Triton program."""
    ps = torch.tensor(
        [1.0 - i / grid for i in range(total_steps)],
        dtype=min_val.dtype,
        device=min_val.device,
    )
    best_p = ps[best_step.long()]
    return min_val * best_p, max_val * best_p


def grid_search_triton(
    observed: torch.Tensor,
    args: QuantizationArgs,
    maxshrink: float,
    patience: int,
    grid: float,
    norm: float,
    expand: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MSE grid search on the GPU.

    :param observed: (num_observations, rows, groups, group_size)
    :param patience: consecutive non-improving steps before this group stops.
        Counted per group, where eager counts once for the whole tensor.
    :param expand: factor applied to the observed min/max before the search,
        same as eager. The shrink grid then runs from that widened range.
    :return: the min/max range minimising the quantization error per group
    """
    import math

    min_val = (torch.amin(observed, dim=(0, -1)) * expand).contiguous()
    max_val = (torch.amax(observed, dim=(0, -1)) * expand).contiguous()
    total_steps = int(maxshrink * grid) if grid > 0 else 0
    if total_steps < 1:
        # nothing to search. eager returns the unshrunk range here, and
        # 1.0 / grid would divide by zero for grid == 0
        return min_val, max_val

    # Calculate the base qparams once. The kernel evaluates scale * p per
    # step, and the zero point is p-invariant in exact arithmetic
    # (min / scale does not change when min and max shrink together).
    scale, zero_point = calculate_qparams(
        min_vals=min_val,
        max_vals=max_val,
        quantization_args=args,
        global_scale=None,
    )
    scale = scale.to(torch.float32).contiguous()
    zero_point = zero_point.to(torch.float32).contiguous()

    codes = _codebook(args).to(observed.device)
    num_codes = codes.numel()

    _, num_rows, num_groups, group_size = observed.shape
    observed = observed.contiguous()
    best_step = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )

    _grid_search_kernel[(num_rows * num_groups,)](
        observed,
        scale,
        zero_point,
        codes,
        best_step,
        num_rows,
        num_groups,
        group_size,
        total_steps,
        num_codes,
        observed.stride(1),
        observed.stride(2),
        # eager checks its counter only after incrementing it, so the value
        # is at least 1 by then and patience=0 behaves exactly like 1. The
        # kernel checks before evaluating, so a raw 0 would search nothing.
        # The observer kwarg accepts 0, so this is a value users do pass.
        1.0 / grid,
        float(norm),
        max(int(patience), 1),
        BLOCK_G=triton.next_power_of_2(group_size),
        LOG_C=max(1, math.ceil(math.log2(num_codes))),
        TOTAL_STEPS=triton.next_power_of_2(total_steps),
        NUM_NEIGHBORS=neighbors_for_config(args, grid, maxshrink),
    )

    return _restore_range(min_val, max_val, best_step.long(), grid, total_steps)
