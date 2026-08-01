"""Triton grid search for the MSE observer.

The eager search fake-quantizes the whole tensor once per shrink step, so it
reads and writes ``observed`` twenty times over. This keeps each group's
values in registers for the whole search and walks the codebook incrementally
instead: shrinking the scale by a fraction of a percent moves most values to
the same code they already had, so after the first step each value only has
to compare against a bounded number of neighbors.

What that bound is, and where it stops holding, is the subtle part; see
:func:`neighbors_for_config`.
"""

from typing import Tuple

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationType
from compressed_tensors.quantization.utils.helpers import calculate_range

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the fallback test
    _TRITON_AVAILABLE = False

__all__ = ["can_use_triton", "grid_search_triton"]

# Rounding of the error accumulation, as a kernel constant: 0 leaves it in
# fp32, 1 and 2 round to bf16 and fp16 after every step. Production
# _calculate_error runs in the observed dtype, and reproducing that is what
# makes the two agree - computing in fp32 instead disagrees with eager on
# 583 of 2048 groups for bf16 int8, at up to 84% relative regret.
_ROUND_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}

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
        return torch.arange(
            -(2 ** (bits - 1)), 2 ** (bits - 1), dtype=torch.float32
        )
    if args.type == QuantizationType.FLOAT and bits == 8:
        codes = (
            torch.arange(256, dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .float()
        )
        return codes[torch.isfinite(codes)].unique().sort().values
    return None


def _tie_parity(codes: torch.Tensor) -> int:
    """Which index wins when two codes are equidistant.

    Every rounding path in compressed_tensors breaks ties toward the even
    mantissa - torch.round for int, the fp8 cast for float8 - which on a
    sorted codebook is a parity rule on the index, fixed by where zero sits.
    Taking the lower code unconditionally gets half of them wrong.
    """
    hits = (codes.detach().cpu() == 0).nonzero()
    return (int(hits[0]) % 2) if hits.numel() else 0


def _half_bit_range(args: QuantizationArgs) -> float:
    """The divisor calculate_qparams uses for a symmetric scale."""
    q_min, q_max = calculate_range(args, torch.device("cpu"))
    return (float(q_max) - float(q_min)) / 2.0


def neighbors_for_config(
    args: QuantizationArgs, grid: float, maxshrink: float, dtype: torch.dtype
) -> int:
    """How many neighbors a value can move in one step of the search.

    Between consecutive steps the scale shrinks by ``s = p_{i+1} / p_i``, so a
    value sitting at ``cutoff[i]`` crosses ``cutoff[i+k]`` only when
    ``cutoff[i] / cutoff[i+k] >= s``. The answer is the smallest ``k`` for
    which that ratio stays below ``s`` for every cutoff.

    Two details make the plain ratio too optimistic:

    The scale is rounded twice, after ``min/max * p`` and again after dividing
    by the bit range, and each rounding can move a value half an ulp. Two
    consecutive scales can therefore show a ratio as bad as ``s * f**2`` with
    ``f = (1 - eps/2) / (1 + eps/2)``. Enumerating every representable bf16
    and fp16 scale, modelling a single rounding falls under the real worst
    case while the squared form covers it.

    That reasoning is relative, so it does not survive into the subnormal
    range, where the spacing is absolute and a scale can halve in one step -
    the same sweep finds a worst ratio of exactly 0.5 for every format, which
    would mean 64 neighbors for bf16 int8. Sizing N for that would punish
    every normal input, so the kernel instead runs its full binary search for
    any step whose scale is subnormal, and N stays a function of the config.
    Keeping it config-only also matters because ``NUM_NEIGHBORS`` is a
    ``tl.constexpr``: deriving it from the data would compile a separate
    kernel per layer.
    """
    codes = _codebook(args)
    if codes is None:
        return 1
    cutoffs = (codes[:-1] + codes[1:]) * 0.5

    total_steps = int(maxshrink * grid)
    ratio = (grid - total_steps) / (grid - total_steps + 1)
    if dtype != torch.float32:
        slack = float(torch.finfo(dtype).eps) / 2.0
        ratio *= ((1.0 - slack) / (1.0 + slack)) ** 2

    limit = int(cutoffs.numel())
    worst = 1
    for side in (cutoffs[cutoffs > 0], -cutoffs[cutoffs < 0]):
        side = side.double().sort().values
        if side.numel() < 2:
            continue
        found = limit
        for k in range(1, side.numel()):
            if (side[:-k] / side[k:]).max().item() < ratio:
                found = k
                break
        worst = max(worst, found)
    return min(worst, limit)


def can_use_triton(observed: torch.Tensor, args: QuantizationArgs) -> bool:
    """Whether the kernel reproduces the eager search for this input.

    Deliberately narrow. Everything outside this keeps the existing eager or
    compiled path, which stays the reference:

    - ``observed`` must be 4d. TENSOR strategy flattens to 3d, and the
      launcher reads (obs, rows, groups, group_size).
    - symmetric only: the codebook search has no zero point.
    - int4, int8 and float8 only. float4 rounds through bfloat16 inside
      compressed_tensors before applying its thresholds, so a nearest-code
      search disagrees on ~5% of groups in fp32 and fp16, and wider int
      widths have not been checked at all.
    - ``scale_dtype`` unset. Rounding the scale, or deriving it as an MX
      exponent, is a different scale path from the symmetric formula the
      kernel reconstructs; NVFP4 additionally applies a global scale that
      makes an exact match unreachable.
    - the group has to fit a block the kernel has been run at.
    - CUDA proper. torch.Tensor.is_cuda is also True under ROCm, and the
      error path calls into libdevice.
    """
    return (
        _TRITON_AVAILABLE
        and observed.is_cuda
        and torch.version.hip is None
        and observed.ndim == 4
        and observed.shape[-1] <= _MAX_GROUP_SIZE
        and bool(args.symmetric)
        and args.scale_dtype is None
        and observed.dtype in _ROUND_CODE
        and _codebook(args) is not None
    )


if _TRITON_AVAILABLE:

    @triton.jit
    def _round_to(x, ROUND_TO: tl.constexpr):
        """Round an fp32 value to the observed dtype and back."""
        if ROUND_TO == 1:
            return x.to(tl.bfloat16).to(tl.float32)
        elif ROUND_TO == 2:
            return x.to(tl.float16).to(tl.float32)
        return x

    @triton.jit
    def _grid_search_kernel(
        observed_ptr,
        min_val_ptr,
        max_val_ptr,
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
        BLOCK_G: tl.constexpr,
        LOG_C: tl.constexpr,
        TOTAL_STEPS: tl.constexpr,
        NUM_NEIGHBORS: tl.constexpr,
        TIE_PARITY: tl.constexpr,
        ROUND_TO: tl.constexpr,
        HALF_RANGE: tl.constexpr,
        TINY_NORMAL: tl.constexpr,
        ZERO_SCALE: tl.constexpr,
    ):
        """One program per group; the group's values stay in registers.

        The scale is rebuilt each step from min/max rather than handed in
        precomputed. Eager shrinks min/max first, in the observed dtype, and
        only then calls calculate_qparams; multiplying one fp32 scale by p
        skips that rounding and costs 345 of 2048 groups on bf16 int8.
        """
        pid = tl.program_id(0)
        row = pid // num_groups
        group = pid % num_groups
        if row >= num_rows:
            return

        g_idx = tl.arange(0, BLOCK_G)
        g_mask = g_idx < group_size
        obs = tl.load(
            observed_ptr + row * stride_obs_row + group * stride_obs_group
            + g_idx,
            mask=g_mask,
            other=0.0,
        ).to(tl.float32)

        min_val = tl.load(min_val_ptr + row * num_groups + group)
        max_val = tl.load(max_val_ptr + row * num_groups + group)
        ig = inv_grid.to(tl.float32)
        norm_f = norm.to(tl.float32)

        shift_dir = tl.where(obs >= 0, 1, -1)
        best_err = tl.full([], float("inf"), dtype=tl.float32)
        best_s = 0
        bin_idx = tl.zeros([BLOCK_G], dtype=tl.int32)

        for step in range(TOTAL_STEPS):
            if step < total_steps:
                p = (1.0 - step * ig).to(tl.float32)
                # calculate_qparams(min * p, max * p), symmetric branch, with
                # both roundings eager performs: the shrink lands in the
                # observed dtype because min/max come from amin/amax on it,
                # and so does the quotient, since calculate_qparams divides a
                # tensor still in that dtype by a python float.
                smin = tl.minimum(
                    _round_to(min_val.to(tl.float32) * p, ROUND_TO), 0.0
                )
                smax = tl.maximum(
                    _round_to(max_val.to(tl.float32) * p, ROUND_TO), 0.0
                )
                raw_scale = _round_to(
                    tl.maximum(tl.abs(smin), tl.abs(smax)) / HALF_RANGE,
                    ROUND_TO,
                )
                # calculate_qparams replaces an exactly zero scale with the
                # dtype's eps rather than clamping small ones, so match that
                # instead of flooring everything at some tiny constant
                eff_scale = tl.where(raw_scale == 0.0, ZERO_SCALE,
                                     raw_scale).to(tl.float32)

                # fake_quantize rounds x/scale to the observed dtype before
                # applying its thresholds
                sel = _round_to(obs / eff_scale, ROUND_TO)

                # A subnormal scale voids the neighbor bound, so search the
                # whole codebook for those steps rather than sizing N for a
                # range that barely occurs.
                if step == 0 or eff_scale < TINY_NORMAL:
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
                    left_wins = (d_left < d_right) | (
                        (d_left == d_right) & ((idx_left % 2) == TIE_PARITY)
                    )
                    bin_idx = tl.where(left_wins, idx_left, idx_right)
                else:
                    new_bin = bin_idx
                    new_d = tl.abs(sel - tl.load(codes_ptr + bin_idx))
                    for k in tl.static_range(1, NUM_NEIGHBORS + 1):
                        idx_k = tl.maximum(
                            tl.minimum(
                                bin_idx + shift_dir * k, num_codes - 1
                            ),
                            0,
                        )
                        d_k = tl.abs(sel - tl.load(codes_ptr + idx_k))
                        better = (d_k < new_d) | (
                            (d_k == new_d) & ((idx_k % 2) == TIE_PARITY)
                        )
                        new_bin = tl.where(better, idx_k, new_bin)
                        new_d = tl.where(better, d_k, new_d)
                    bin_idx = new_bin

                # _calculate_error, step for step: dequantize, round, subtract
                # in the observed dtype, raise to an exponent that was itself
                # cast, round each power, accumulate in fp32, round the sum.
                q = _round_to(tl.load(codes_ptr + bin_idx) * eff_scale,
                              ROUND_TO)
                resid = tl.abs(_round_to(q - obs, ROUND_TO))
                powed = _round_to(
                    tl.extra.cuda.libdevice.pow(resid, norm_f), ROUND_TO
                )
                err = _round_to(
                    tl.sum(tl.where(g_mask, powed, 0.0), axis=0), ROUND_TO
                )

                is_better = err < best_err
                best_err = tl.where(is_better, err, best_err)
                best_s = tl.where(is_better, step, best_s)

        tl.store(best_step_ptr + row * num_groups + group, best_s)


def _restore_range(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    best_step: torch.Tensor,
    grid: float,
    total_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rebuild the winning range the way eager produced it.

    ``min_val * ps[best_step]`` with ``ps`` materialised in the storage dtype
    would round p before the multiply, which is not eager's arithmetic in
    bf16. Walking the steps and overwriting only where each one won keeps the
    python float multiply, at one transient tensor.
    """
    best_min = torch.empty_like(min_val)
    best_max = torch.empty_like(max_val)
    for i in range(total_steps):
        won = best_step == i
        p = 1.0 - i / grid
        best_min = torch.where(won, min_val * p, best_min)
        best_max = torch.where(won, max_val * p, best_max)
    return best_min, best_max


def grid_search_triton(
    observed: torch.Tensor,
    args: QuantizationArgs,
    maxshrink: float,
    grid: float,
    norm: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full-grid MSE search on the GPU.

    Early stopping is not implemented here on purpose. Patience is a
    heuristic on a global counter that resets whenever any group improves,
    which does not carry over to a kernel where each program only sees its
    own group; a per-group version is a different knob and measurably worse
    at 4 bits. Searching the whole grid keeps the result identical to eager
    and is still far cheaper, since the cost the kernel removes is the
    repeated pass over ``observed``, not the step count.

    :param observed: (num_observations, rows, groups, group_size)
    :return: the min/max range minimising the quantization error per group
    """
    import math

    min_val = torch.amin(observed, dim=(0, -1)).contiguous()
    max_val = torch.amax(observed, dim=(0, -1)).contiguous()
    total_steps = int(maxshrink * grid)

    codes = _codebook(args)
    parity = _tie_parity(codes)
    codes = codes.to(observed.device)
    num_codes = codes.numel()

    _, num_rows, num_groups, group_size = observed.shape
    observed = observed.contiguous()
    best_step = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )

    # torch casts a python-float exponent to the tensor dtype, so bf16 turns
    # a norm of 2.4 into 2.40625; the kernel has to use the same value.
    exponent = (
        float(norm)
        if observed.dtype == torch.float32
        else float(torch.tensor(norm, dtype=observed.dtype).item())
    )

    _grid_search_kernel[(num_rows * num_groups,)](
        observed, min_val, max_val, codes, best_step,
        num_rows, num_groups, group_size, total_steps, num_codes,
        observed.stride(1), observed.stride(2),
        1.0 / grid, exponent,
        BLOCK_G=triton.next_power_of_2(group_size),
        LOG_C=max(1, math.ceil(math.log2(num_codes))),
        TOTAL_STEPS=triton.next_power_of_2(total_steps),
        NUM_NEIGHBORS=neighbors_for_config(
            args, grid, maxshrink, observed.dtype
        ),
        TIE_PARITY=parity,
        ROUND_TO=_ROUND_CODE[observed.dtype],
        HALF_RANGE=_half_bit_range(args),
        # the subnormal fallback applies in every dtype: fp32 has a
        # subnormal range too, and the relative bound is just as void there
        TINY_NORMAL=float(torch.finfo(observed.dtype).tiny),
        ZERO_SCALE=float(torch.finfo(observed.dtype).eps),
    )

    return _restore_range(
        min_val, max_val, best_step.long(), grid, total_steps
    )
