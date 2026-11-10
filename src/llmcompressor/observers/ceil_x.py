"""
Ceil_x observer for MXFP4 per-block weight-only quantization.

Implements the ceil_x algorithm: uses ``ceil(log2(max(|x|)/c))``
instead of the traditional ``floor(log2(max(|x|)))`` to compute the shared
exponent, with a configurable divisor *c* that controls the range/resolution
trade-off. An optional MSE search selects the best *c* from a candidate grid.

The observer produces ``min_vals`` / ``max_vals`` that, when fed into
``calculate_qparams`` for a FLOAT type with TENSOR_GROUP strategy, yield
the ceil_x shared exponent. The trick: set
``adjusted_absmax = 2^(ceil(log2(absmax/c + eps)) + e_max)`` so that
``floor(log2(adjusted_absmax)) == ceil(log2(absmax/c + eps)) + e_max``,
which cancels the ``- e_max`` inside ``generate_mx_scales`` to yield
``E8M0 = 127 + ceil(log2(absmax/c + eps))``.

Usage::

    QuantizationModifier(
        targets="Linear",
        scheme="MXFP4",
        weight_observer="ceil_x",
        ignore=["lm_head"],
    )

Or with MSE search::

    QuantizationModifier(
        targets="Linear",
        scheme="MXFP4A16",
        weight_observer="ceil_x",
        observer_kwargs={
            "enable_search": True,
            "search_min": 6.0,
            "search_max": 12.0,
            "search_step": 0.25,
        },
        ignore=["lm_head"],
    )
"""

from typing import Optional

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams

from llmcompressor.observers.base import MinMaxTuple, Observer

__all__ = ["CeilXObserver", "MXFP4CeilXObserver"]


# MXFP4 format constants (E2M1, block_size=32)
_MXFP4_EBITS = 2
_MXFP4_EMAX = 2 ** (_MXFP4_EBITS - 1)  # = 2
_MXFP4_SCALE_BITS = 8  # E8M0 shared exponent
_MXFP4_SCALE_EMAX = 2 ** (_MXFP4_SCALE_BITS - 1) - 1  # = 127
_EPS = 9.6e-7


def _ceil_x_adjust(
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    ceil_x_value: float,
) -> MinMaxTuple:
    """Adjust min/max so that ``calculate_qparams`` produces the ceil_x exponent.

    For symmetric quantization the shared exponent is derived from
    ``max(|x|)``. Setting
    ``adjusted = 2^(ceil(log2(max/c + eps)) + e_max)`` makes
    ``floor(log2(adjusted)) == ceil(log2(max/c + eps)) + e_max``,
    which cancels the ``- e_max`` in ``generate_mx_scales`` to yield
    ``E8M0 = 127 + ceil(log2(max/c + eps))``.

    This function is only valid for symmetric quantization. The caller is
    responsible for ensuring the quantization args specify symmetric=True.

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


def _ceil_x_search_best(
    observed: torch.Tensor,
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    args: QuantizationArgs,
    search_min: float = 6.0,
    search_max: float = 12.0,
    search_step: float = 0.25,
) -> float:
    """Search for the best ``ceil_x_value`` that minimizes MSE.

    For each candidate *c*, compute adjusted min/max, derive qparams,
    fake-quantize the observed values, and measure reconstruction MSE.
    Return the candidate with the lowest error.

    :param observed: flattened observations of shape
                     ``(num_observations, *qparam_shape, group_size)``
    :param min_vals: per-group minimum values
    :param max_vals: per-group maximum values
    :param args: quantization arguments
    :param search_min: lower bound of search range
    :param search_max: upper bound of search range
    :param search_step: step size between candidates
    :return: best ceil_x_value found
    """
    num_steps = int(round((search_max - search_min) / search_step)) + 1
    candidates = [search_min + i * search_step for i in range(num_steps)]

    best_mse = float("inf")
    best_value = search_min

    # For TENSOR_GROUP with scale_dtype, temporarily clear scale_dtype so
    # calculate_qparams works in FP32 (the global_scale cancels out in MSE)
    effective_args = args
    if (
        args.strategy == QuantizationStrategy.TENSOR_GROUP
        and args.scale_dtype is not None
    ):
        effective_args = args.model_copy(update={"scale_dtype": None})

    for value in candidates:
        adj_min, adj_max = _ceil_x_adjust(min_vals, max_vals, value)
        scale, zero_point = calculate_qparams(
            min_vals=adj_min,
            max_vals=adj_max,
            quantization_args=effective_args,
            global_scale=None,
        )
        # fake_quantize expects scale/zero_point broadcastable to observed
        # observed shape: (num_obs, *qparam_shape, group_size)
        # scale shape: (*qparam_shape,) → need unsqueeze(0) and unsqueeze(-1)
        q_scale = scale.unsqueeze(0).unsqueeze(-1)
        q_zp = zero_point.unsqueeze(0).unsqueeze(-1)
        quantized = fake_quantize(observed, q_scale, q_zp, effective_args)
        mse = ((quantized - observed) ** 2).mean().item()

        if mse < best_mse:
            best_mse = mse
            best_value = value

    return best_value


@Observer.register("ceil_x")
class CeilXObserver(Observer):
    """
    Ceil_x observer for MXFP4 per-block weight-only quantization.

    Uses ``ceil(log2(max(|x|)/c))`` to compute the shared exponent, where *c*
    is a configurable divisor (default 7.25, range [6.0, 12.0]). This reduces
    the large-value truncation that occurs with traditional floor-based
    scaling in MXFP4, trading truncation of 1-2 outlier values per block for
    a halved quantization step size on the remaining values.

    The observer adjusts ``min_vals`` / ``max_vals`` so that the downstream
    ``calculate_qparams`` produces the ceil_x exponent. For TENSOR_GROUP
    strategy (MXFP4), the base class handles ``global_scale`` computation
    and fusion across fused layer groups (Q/K/V, gate/up).

    Optional MSE search: when ``enable_search=True``, the observer enumerates
    ``ceil_x_value`` candidates in ``[search_min, search_max]`` with step
    ``search_step`` and selects the value that minimizes weight reconstruction
    MSE.

    Observer kwargs:
        ceil_x_value: divisor *c* (default 7.25, range [6.0, 12.0])
        enable_search: whether to search for the best *c* (default False)
        search_min: search range lower bound (default 6.0)
        search_max: search range upper bound (default 12.0)
        search_step: search step size (default 0.25)
    """

    _act_sync_dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.args.observer_kwargs
        self.ceil_x_value = kw.get("ceil_x_value", 7.25)
        self.enable_search = kw.get("enable_search", False)
        self.search_min = kw.get("search_min", 6.0)
        self.search_max = kw.get("search_max", 12.0)
        self.search_step = kw.get("search_step", 0.25)

        # Validate format constraints: ceil_x only supports MXFP4 symmetric
        if self.args.type != QuantizationType.FLOAT:
            raise ValueError(
                f"ceil_x only supports FLOAT type (MXFP4), got {self.args.type}"
            )
        if not self.args.symmetric:
            raise ValueError("ceil_x only supports symmetric quantization")
        if self.args.strategy != QuantizationStrategy.TENSOR_GROUP:
            raise ValueError(
                f"ceil_x only supports TENSOR_GROUP strategy, "
                f"got {self.args.strategy}"
            )

        # Validate ceil_x_value range
        if not (6.0 <= self.ceil_x_value <= 12.0):
            raise ValueError(
                f"ceil_x_value must be in [6.0, 12.0], got {self.ceil_x_value}"
            )
        if self.enable_search:
            if not (6.0 <= self.search_min <= 12.0) or not (
                6.0 <= self.search_max <= 12.0
            ):
                raise ValueError(
                    f"search_min and search_max must be in [6.0, 12.0], "
                    f"got [{self.search_min}, {self.search_max}]"
                )
            if self.search_max <= self.search_min:
                raise ValueError(
                    f"search_max ({self.search_max}) must be greater than "
                    f"search_min ({self.search_min})"
                )
            if self.search_step <= 0:
                raise ValueError(
                    f"search_step must be positive, got {self.search_step}"
                )

        # Store the effective value (may be updated by search)
        self._effective_ceil_x: Optional[float] = None

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        # Step 1: compute raw per-group min/max
        # observed shape: (num_observations, *qparam_shape, group_size)
        # Reduce over dim 0 (observations) and dim -1 (group_size)
        raw_min = torch.amin(observed, dim=(0, -1))
        raw_max = torch.amax(observed, dim=(0, -1))

        # Step 2: optionally search for the best ceil_x_value
        if self.enable_search:
            self._effective_ceil_x = _ceil_x_search_best(
                observed,
                raw_min,
                raw_max,
                self.args,
                search_min=self.search_min,
                search_max=self.search_max,
                search_step=self.search_step,
            )
        else:
            self._effective_ceil_x = self.ceil_x_value

        # Step 3: adjust min/max for ceil_x shared exponent
        self.min_vals, self.max_vals = _ceil_x_adjust(
            raw_min, raw_max, self._effective_ceil_x
        )


@Observer.register("mxfp4_ceil_x")
class MXFP4CeilXObserver(CeilXObserver):
    """
    Alias for :class:`CeilXObserver` for use with MXFP4 schemes.

    This class has the same defaults as :class:`CeilXObserver` (no tuning
    beyond the base class). It is provided as a descriptive name so that
    recipes can make the MXFP4 intent explicit. The default
    ``ceil_x_value=7.25`` is empirically optimal for MXFP4 symmetric
    weight-only quantization.

    Usage::

        QuantizationModifier(
            targets="Linear",
            scheme="MXFP4A16",
            weight_observer="mxfp4_ceil_x",
            ignore=["lm_head"],
        )
    """

    pass
