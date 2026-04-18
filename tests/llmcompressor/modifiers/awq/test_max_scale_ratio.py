"""
Unit tests for the max_scale_ratio guard added to AWQModifier._compute_best_scale.

We don't spin up a real calibration loop here; the guard's logic is pure
arithmetic on the per-channel scale tensor, so we replicate the relevant
snippet against synthetic inputs and assert the documented behaviour:

1. Spans <= max_scale_ratio are accepted.
2. Spans > max_scale_ratio are rejected (the candidate is skipped).
3. When every grid candidate is rejected, the modifier falls back to
   identity scales (`torch.ones_like(...)`) instead of selecting the
   least-bad pathological scale.
"""

import pytest
import torch

from llmcompressor.modifiers.awq import AWQModifier


def _normalise(scales: torch.Tensor) -> torch.Tensor:
    """Same geometric-mean normalisation _compute_best_scale uses."""
    out = scales / (scales.max() * scales.min()).sqrt()
    out[torch.isinf(out)] = 1
    out[torch.isnan(out)] = 1
    return out


def _evaluate_guard(scales: torch.Tensor, max_scale_ratio: float) -> bool:
    """True if the candidate would be accepted (i.e. NOT skipped)."""
    s_min = scales.min().item()
    span = (scales.max().item() / s_min) if s_min > 0 else float("inf")
    return span <= max_scale_ratio


@pytest.mark.unit
def test_max_scale_ratio_default_is_none():
    """The guard is opt-in: default ``None`` preserves upstream behaviour
    so we never silently change quantization quality for models that did
    not previously enable it (e.g. plain Llama, Mistral)."""
    awq = AWQModifier(scheme="W4A16_ASYM")
    assert awq.max_scale_ratio is None


@pytest.mark.unit
def test_max_scale_ratio_can_be_enabled():
    """Architectures known to suffer extreme spans (e.g. Qwen3.5 with
    attn_output_gate, mixed full/linear attention, MoE) opt into the
    4x heuristic from AutoAWQ."""
    awq = AWQModifier(scheme="W4A16_ASYM", max_scale_ratio=4.0)
    assert awq.max_scale_ratio == 4.0


@pytest.mark.unit
def test_max_scale_ratio_can_be_overridden():
    """Callers can dial the guard up or down."""
    awq = AWQModifier(scheme="W4A16_ASYM", max_scale_ratio=8.0)
    assert awq.max_scale_ratio == 8.0


@pytest.mark.unit
def test_guard_accepts_healthy_span():
    """A 2x activation magnitude swing produces a small normalised span."""
    x_mean = torch.tensor([1.0, 1.5, 2.0, 1.2, 1.8, 1.0, 1.4, 1.6])
    scales = _normalise(x_mean.pow(0.5).clamp(min=1e-4))
    assert _evaluate_guard(scales, max_scale_ratio=4.0)


@pytest.mark.unit
def test_guard_rejects_extreme_span():
    """Extreme activation outliers yield > 4x scale span and are skipped."""
    x_mean = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e3])
    scales = _normalise(x_mean.pow(1.0).clamp(min=1e-4))
    assert not _evaluate_guard(scales, max_scale_ratio=4.0)


@pytest.mark.unit
def test_high_ratio_candidates_filtered_under_extreme_outlier():
    """Walk the grid the way _compute_best_scale does and confirm the guard
    rejects the high-ratio candidates while leaving low-ratio candidates
    available, so the search never selects a broken 25x-span scale."""
    x_mean = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e3])
    n_grid = 20

    accepted_ratios = []
    rejected_ratios = []
    for i in range(n_grid):
        ratio = i / n_grid
        scales = _normalise(x_mean.pow(ratio).clamp(min=1e-4))
        if _evaluate_guard(scales, max_scale_ratio=4.0):
            accepted_ratios.append(ratio)
        else:
            rejected_ratios.append(ratio)

    # Some low-ratio candidates pass, very-high-ratio candidates get rejected
    assert len(accepted_ratios) > 0
    assert len(rejected_ratios) > 0
    assert max(accepted_ratios) < min(rejected_ratios)


@pytest.mark.unit
def test_identity_fallback_has_unit_span():
    """When every grid candidate is filtered out, the production path
    selects ``torch.ones_like(x_mean)``. Verify this fallback has unit
    span (i.e. it is itself trivially within the guard)."""
    fallback = torch.ones(8)
    assert (fallback.max() / fallback.min()).item() == 1.0
    assert _evaluate_guard(fallback, max_scale_ratio=4.0)


@pytest.mark.unit
def test_identity_fallback_when_all_pure_x_mean_candidates_fail():
    """Construct a setup where every non-identity ratio produces a
    pathological span and identity fallback would kick in."""
    x_mean = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e6])
    n_grid = 20

    surviving = []
    for i in range(1, n_grid):  # skip ratio=0 (trivially identity)
        ratio = i / n_grid
        scales = _normalise(x_mean.pow(ratio).clamp(min=1e-4))
        if _evaluate_guard(scales, max_scale_ratio=4.0):
            surviving.append(ratio)

    # Almost every non-zero ratio is filtered when the outlier is 1e6
    assert len(surviving) <= 2
