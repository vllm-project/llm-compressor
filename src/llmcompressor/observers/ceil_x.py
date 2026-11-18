"""
Ceil_x observer for MXFP4 per-block weight-only quantization.

Implements the ceil_x algorithm: uses ``ceil(log2(max(|x|)/c + eps))``
instead of the min-max exponent derived by ``generate_mx_scales`` to
compute the MXFP4 shared exponent. The divisor *c* is fixed at ``7.25``:
it does not eliminate large-value truncation (that requires ``c <= 6.0``)
but achieves the best truncation-vs-resolution trade-off, sacrificing
1-2 outlier values per 32-value block to truncation while halving the
quantization step size for the remaining values.

The observer produces ``min_vals`` / ``max_vals`` that, when fed into
``calculate_qparams`` for a FLOAT type with GROUP strategy (the MXFP4
scheme), yield the ceil_x shared exponent. The trick: set
``adjusted_absmax = 2^(ceil(log2(absmax/c + eps)) + e_max)`` so that
``floor(log2(adjusted_absmax)) == ceil(log2(absmax/c + eps)) + e_max``,
which cancels the ``- e_max`` inside ``generate_mx_scales`` to yield
``E8M0 = 127 + ceil(log2(absmax/c + eps))``.

A single observer is provided, specific to MXFP4 (E2M1 elements with
32-value blocks and E8M0 shared exponents):

- ``mxfp4_ceil_x``: :class:`MemorylessMSEObserver` subclass that runs the
  shared MSE grid search from the ceil_x adjusted range, selecting
  per-group ranges that minimize reconstruction error.

Usage::

    QuantizationModifier(
        targets="Linear",
        scheme="MXFP4A16",
        weight_observer="mxfp4_ceil_x",
        ignore=["lm_head"],
    )

Task-level results for Qwen3-14B are recorded in the Open Research Hub:
``docs/research/mxfp4-ceilx-qwen3-14b-yzmkbk.md``.
"""

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.mxfp_utils import (
    should_generate_mx_scales,
)

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.grid_search import _grid_search_observer
from llmcompressor.observers.mse import MemorylessMSEObserver

__all__ = ["MXFP4CeilXObserver"]


# MXFP4 format constants (E2M1, block_size=32)
_MXFP4_EBITS = 2
_MXFP4_EMAX = 2 ** (_MXFP4_EBITS - 1)  # = 2
_MXFP4_SCALE_BITS = 8  # E8M0 shared exponent
_MXFP4_SCALE_EMAX = 2 ** (_MXFP4_SCALE_BITS - 1) - 1  # = 127
_EPS = 9.6e-7

# Fixed ceil_x divisor. c=7.25 is empirically optimal for MXFP4 weight
# quantization; the supported range [6.0, 12.0] was explored offline. The
# divisor is intentionally not a user-facing option: it is specific to the
# E2M1 element format and easy to misconfigure.
_MXFP4_CEIL_X_VALUE = 7.25


def _validate_mxfp4_args(args: QuantizationArgs) -> None:
    """Validate that the quantization args describe MXFP4 weights.

    ceil_x only applies to the MXFP4 format: the divisor constant is tuned
    for E2M1 elements, and the min/max adjustment trick assumes 32-value
    blocks with E8M0 shared exponents.

    :param args: quantization args to validate
    :raises ValueError: if any format constraint is violated
    """
    if args.type != QuantizationType.FLOAT:
        raise ValueError(
            f"mxfp4_ceil_x only supports FLOAT type (MXFP4), got {args.type}"
        )
    if args.num_bits != 4:
        raise ValueError(
            f"mxfp4_ceil_x only supports 4-bit (E2M1), got {args.num_bits} bits"
        )
    if not args.symmetric:
        raise ValueError("mxfp4_ceil_x only supports symmetric quantization")
    if args.strategy != QuantizationStrategy.GROUP:
        raise ValueError(
            f"mxfp4_ceil_x only supports the GROUP strategy used by the MXFP4 "
            f"scheme, got {args.strategy}"
        )
    if not should_generate_mx_scales(args):
        raise ValueError(
            "mxfp4_ceil_x requires MXFP4 scale generation (group_size=32 with "
            f"scale_dtype=uint8), got group_size={args.group_size}, "
            f"scale_dtype={args.scale_dtype}"
        )


def _mxfp4_ceil_x_adjust(
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    ceil_x_value: float = _MXFP4_CEIL_X_VALUE,
) -> MinMaxTuple:
    """Adjust min/max so that ``calculate_qparams`` produces the ceil_x exponent.

    For symmetric quantization the shared exponent is derived from
    ``max(|x|)``. Setting
    ``adjusted = 2^(ceil(log2(max/c + eps)) + e_max)`` makes
    ``floor(log2(adjusted)) == ceil(log2(max/c + eps)) + e_max``,
    which cancels the ``- e_max`` in ``generate_mx_scales`` to yield
    ``E8M0 = 127 + ceil(log2(max/c + eps))``.

    This function is only valid for symmetric MXFP4 quantization. The caller
    is responsible for ensuring the quantization args describe that format.

    :param min_vals: per-group minimum values
    :param max_vals: per-group maximum values
    :param ceil_x_value: divisor *c*, controls the exponent jump boundary
    :return: adjusted (min_vals, max_vals)
    """
    abs_min = min_vals.abs()
    abs_max = max_vals.abs()
    absmax = torch.maximum(abs_min, abs_max)

    # Early return for empty tensors (e.g. ignored modules)
    if absmax.numel() == 0:
        return min_vals, max_vals

    # ceil_x core: shared_exp = ceil(log2(max(|x|) / c + eps))
    # (raw ceil value; the -e_max offset is handled via +_MXFP4_EMAX below)
    #
    # The log2/ceil computation runs in bfloat16 (not float32) so that log2
    # values just below an integer (e.g. -6.9936) round to the exact integer
    # (-7.0) under bfloat16's 8-bit mantissa, causing ceil() to return -7
    # instead of -6.  This matches the empirical tuning of c=7.25, which was
    # determined under bfloat16 precision.  In float32 the fractional part is
    # preserved and ceil() returns -6, producing a 2x larger scale for ~2% of
    # groups and degrading accuracy.
    orig_dtype = absmax.dtype
    bf16_absmax = absmax.to(torch.bfloat16)
    log2_input = (bf16_absmax / ceil_x_value) + _EPS
    shared_exp = torch.ceil(torch.log2(log2_input)).to(torch.float32)

    # Clip shared_exp so that shared_exp + _MXFP4_EMAX stays within
    # [-127, 127], the valid range for generate_mx_scales exponent input
    shared_exp = shared_exp.clamp(
        -_MXFP4_SCALE_EMAX - _MXFP4_EMAX,
        _MXFP4_SCALE_EMAX - _MXFP4_EMAX,
    )

    # adjusted_absmax = 2^(shared_exp + _MXFP4_EMAX), which forces
    # generate_mx_scales to produce E8M0 = 127 + shared_exp.
    # generate_mx_scales computes: 127 + floor(log2(scale)) - 2, so we add
    # _MXFP4_EMAX (=2) to shared_exp so that floor(log2(2^(shared_exp+2)))
    # = shared_exp + 2, yielding E8M0 = 127 + shared_exp + 2 - 2 = 127 + shared_exp.
    # Compute in float32 for numerical stability (bfloat16/float16 underflow
    # for exponents below -24), then cast to the original dtype.
    adjusted_absmax = torch.pow(2.0, (shared_exp + _MXFP4_EMAX)).to(orig_dtype)

    # Symmetric: both min and max are set to ±adjusted_absmax
    new_max_vals = adjusted_absmax
    new_min_vals = -adjusted_absmax

    return new_min_vals, new_max_vals


@Observer.register("mxfp4_ceil_x")
class MXFP4CeilXObserver(MemorylessMSEObserver):
    """
    MXFP4 ceil_x observer with per-group MSE grid-search refinement.

    Uses ``ceil(log2(max(|x|)/c + eps))`` with the fixed divisor
    ``c = 7.25`` to compute the MXFP4 shared exponent, then refines it per
    group: the shared MSE grid search
    (see :mod:`llmcompressor.observers.grid_search`) starts from the
    ceil_x adjusted range and selects the per-group candidate that
    minimizes reconstruction error. The first grid candidate reproduces
    the fixed ceil_x exponent itself, so the result is never worse than
    the fixed heuristic under the configured error norm. Setting
    ``maxshrink=0`` (or any ``maxshrink`` that yields an empty grid)
    reproduces the fixed ceil_x observer exactly.

    On MX formats the automatic (exponential) grid halves the range at
    each candidate, which decrements the shared exponent by one: with the
    default ``maxshrink`` the search visits the ceil_x shared exponent
    plus three finer exponents (1/2, 1/4, 1/8) per group, letting each
    block choose its own truncation-vs-resolution trade-off. The grid also
    appends a linear-grid endpoint candidate, but against the power-of-two
    base range it is degenerate, mapping to the 1/8 exponent.

    Inherits the standard MSE observer kwargs (``norm``, ``maxshrink``,
    ``patience``, ``grid``, ...); see :class:`MemorylessMSEObserver`.

    Usage::

        QuantizationModifier(
            targets="Linear",
            scheme="MXFP4A16",
            weight_observer="mxfp4_ceil_x",
            ignore=["lm_head"],
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_mxfp4_args(self.args)
        # Search down to 1/8 of the ceil_x adjusted range. With the
        # exponential grid used for MX scales this visits the ceil_x
        # shared exponent plus three finer exponents (1/2, 1/4, 1/8) per
        # group; the grid's appended linear endpoint is degenerate against
        # the power-of-two base range.
        self.maxshrink = self.args.observer_kwargs.get("maxshrink", 1.0 - 1.0 / 8.0)

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        # observed shape: (num_observations, *qparam_shape, group_size)
        raw_min = torch.amin(observed, dim=(0, -1))
        raw_max = torch.amax(observed, dim=(0, -1))

        # Search from the ceil_x adjusted range instead of the raw
        # observed min/max
        base_min, base_max = _mxfp4_ceil_x_adjust(raw_min, raw_max)
        self.min_vals, self.max_vals = _grid_search_observer(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            expand=self.expand,
            magnitude_normalization=self.magnitude_normalization,
            grid_type=self.grid_type,
            base_range=(base_min, base_max),
        )
