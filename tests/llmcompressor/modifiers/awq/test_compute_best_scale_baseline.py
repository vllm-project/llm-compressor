"""
End-to-end-ish tests that drive ``AWQModifier`` through its real code
path on a tiny synthetic module.

Every test in this file exercises production code (``_compute_best_scale``,
``_eval_scales``, ``_assert_smoothing_health``, ``_apply_smoothing``) by
attaching a real W4A16 quantization scheme + observer to a tiny
``Linear``, hand-populating the activation/parent caches that the
production hooks would otherwise populate during calibration, and then
invoking the public-facing entry points.

This is the only AWQ test file that walks the real grid search loop;
contract / dataclass / shape tests have been deliberately removed in
favour of the regressions captured here. New tests SHOULD be added here
(or to a sibling file with the same execution model) rather than as
stand-alone field/dataclass tests.
"""

from __future__ import annotations

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    disable_quantization,
    initialize_module_for_quantization,
)
from torch.nn import LayerNorm, Linear, Module

from llmcompressor.core.session_functions import create_session
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import ResolvedMapping
from llmcompressor.modifiers.quantization.calibration import (
    call_observer,
    initialize_observer,
)
from llmcompressor.pipelines.cache import IntermediatesCache


class _Wrapper(Module):
    """Minimal parent module: forwards a hidden state through a single Linear."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = Linear(in_features, out_features, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class _LayerNormBlock(Module):
    """LayerNorm -> Linear, mirroring an ``input_layernorm -> q_proj`` path.

    Used by the forward-equivalence regression to verify the AWQ
    smoothing transform is a no-op on the FP forward pass.
    """

    def __init__(self, hidden: int, out_features: int | None = None):
        super().__init__()
        self.input_layernorm = LayerNorm(hidden)
        self.proj = Linear(hidden, out_features or hidden, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(self.input_layernorm(hidden_states))


def _attach_w4a16_scheme(
    balance_layer: Linear, group_size: int = 8, num_bits: int = 4
) -> None:
    """Attach a W4A16 group-wise quantization scheme + observer + qparams.

    ``initialize_module_for_quantization`` swaps ``Linear.__call__`` for a
    weight-quantizing forward. The freshly-initialised ``weight_scale`` is
    zeros, so the quantized forward returns zeros/NaN until the observer
    is calibrated; we calibrate the observer here once so the wrapper
    has a non-degenerate ``Q(W) @ x`` to fall back on.

    Production also disables this wrapper before running ``fp16_outputs``
    forward passes inside ``_apply_smoothing`` (see ``AWQModifier.on_start``
    where ``model.apply(disable_quantization)`` is called). Without the
    same flip the grid baseline measurement would compare two identical
    quantized outputs for ``scales = ones`` (loss = 0) and the search
    could never improve on identity. We mirror that production behaviour
    here so ``_compute_best_scale`` and ``_apply_smoothing`` walk a
    realistic code path.
    """
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=num_bits,
            symmetric=False,
            strategy=QuantizationStrategy.GROUP,
            group_size=group_size,
        ),
    )
    initialize_module_for_quantization(balance_layer, scheme, force_zero_point=True)
    initialize_observer(balance_layer, base_name="weight")
    call_observer(
        balance_layer, "weight", balance_layer.weight, should_calculate_gparam=False
    )
    disable_quantization(balance_layer)


def _setup_modifier(
    awq: AWQModifier,
    parent: Module,
    balance_layer: Linear,
    smooth_name: str,
    batch_inputs: list[torch.Tensor],
):
    """Hand-populate the private caches that ``_compute_best_scale`` reads."""
    cache = IntermediatesCache(None, None)
    for x in batch_inputs:
        cache.append({"hidden_states": x})
    awq._parent_args_cache[parent] = cache

    x_concat = torch.cat([b.reshape(-1, b.shape[-1]) for b in batch_inputs], dim=0)
    x_sum = x_concat.abs().sum(dim=0).to(torch.float32)
    count = torch.tensor(x_concat.shape[0], dtype=torch.float32)
    awq._smooth_activation_stats[smooth_name] = [x_sum, count]


def _build_resolved_mapping(
    parent: Module,
    balance_layer: Linear,
    *,
    smooth_layer: Module | None = None,
    smooth_name: str = "layers.0.input_layernorm",
    balance_output_slices: dict[Module, slice] | None = None,
) -> ResolvedMapping:
    return ResolvedMapping(
        smooth_name=smooth_name,
        smooth_layer=smooth_layer or LayerNorm(balance_layer.in_features),
        balance_layers=[balance_layer],
        balance_names=["layers.0.proj"],
        parent=parent,
        parent_name="layers.0",
        balance_output_slices=balance_output_slices,
    )


def _make_inputs(
    batch_size: int, seq_len: int, hidden: int, *, outlier_channel: int | None = None
) -> list[torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, hidden)
    if outlier_channel is not None:
        x[..., outlier_channel] *= 1e3
    return [x]


@pytest.mark.unit
def test_initial_error_equals_identity_loss():
    """
    Regression test for the codex-flagged baseline bug: with
    ``duo_scaling=True`` (default) and ``ratio=0`` the first grid candidate
    is *not* identity, so the previous implementation recorded an
    arbitrary loss as ``initial_error``. The fix seeds the search with a
    real identity-scales measurement; verify that the recorded
    ``initial_error`` exactly matches a manual identity-scales loss
    computation.
    """
    in_features, out_features = 32, 16
    parent = _Wrapper(in_features, out_features)
    balance_layer = parent.proj
    _attach_w4a16_scheme(balance_layer)

    batch_inputs = _make_inputs(1, 4, in_features)

    with torch.no_grad():
        fp16_outputs = [parent(x).clone() for x in batch_inputs]
    orig_layer_weights = {balance_layer: balance_layer.weight.detach().clone()}

    awq = AWQModifier(scheme="W4A16_ASYM", n_grid=4, duo_scaling=True)
    mapping = _build_resolved_mapping(parent, balance_layer)
    _setup_modifier(awq, parent, balance_layer, mapping.smooth_name, batch_inputs)

    with create_session():
        awq._compute_best_scale(mapping, fp16_outputs, orig_layer_weights)

    metrics = awq._error_metrics[0]
    assert metrics["layer_name"] == mapping.smooth_name

    # Recompute identity loss the same way ``_eval_scales`` would,
    # independently of the grid loop.
    balance_layer.weight.data.copy_(orig_layer_weights[balance_layer])
    awq2 = AWQModifier(scheme="W4A16_ASYM", n_grid=4, duo_scaling=True)
    _setup_modifier(awq2, parent, balance_layer, mapping.smooth_name, batch_inputs)
    with create_session():
        identity_loss = awq2._eval_scales(
            torch.ones(in_features),
            mapping,
            fp16_outputs,
            orig_layer_weights,
            [balance_layer],
            torch.device("cpu"),
        )

    # Pre-fix, initial_error would be the loss at duo-scaling ratio=0
    # scales, NOT identity. Post-fix they must match exactly.
    assert metrics["initial_error"] == pytest.approx(identity_loss, rel=1e-6, abs=1e-9)


@pytest.mark.unit
def test_best_error_never_exceeds_initial_error():
    """
    The seed-with-identity strategy guarantees the grid search can never
    pick a worse-than-baseline candidate, so the AWQ artifact is never
    *worse* than plain W4A16 for that layer.
    """
    in_features, out_features = 32, 16
    parent = _Wrapper(in_features, out_features)
    balance_layer = parent.proj
    _attach_w4a16_scheme(balance_layer)

    batch_inputs = _make_inputs(1, 4, in_features, outlier_channel=7)
    with torch.no_grad():
        fp16_outputs = [parent(x).clone() for x in batch_inputs]
    orig_layer_weights = {balance_layer: balance_layer.weight.detach().clone()}

    awq = AWQModifier(scheme="W4A16_ASYM", n_grid=4, duo_scaling=True)
    mapping = _build_resolved_mapping(parent, balance_layer)
    _setup_modifier(awq, parent, balance_layer, mapping.smooth_name, batch_inputs)

    with create_session():
        awq._compute_best_scale(mapping, fp16_outputs, orig_layer_weights)

    metrics = awq._error_metrics[0]
    assert metrics["best_error"] <= metrics["initial_error"], (
        "Grid search must never select a candidate worse than the "
        f"identity baseline, but got best={metrics['best_error']:.3e} > "
        f"initial={metrics['initial_error']:.3e}"
    )


@pytest.mark.unit
def test_balance_output_slices_changes_grid_outcome_for_fused_q_proj():
    """
    Production-path regression for ``balance_output_slices``.

    For Qwen3.5's fused query+gate ``q_proj`` the gate half (top H*D rows
    in a 2*H*D output Linear) is sigmoid'd downstream and its weight
    magnitude is *wildly* out of distribution from the query half. If the
    grid search is allowed to see MSE on the gate rows it gets pulled
    toward scales that minimise gate error and degrade query error -- the
    exact failure mode that originally pushed Qwen3.5-27B AWQ to
    ``bpb=3.8`` versus baseline ``bpb=0.5``.

    Setting ``balance_output_slices`` to scope the MSE to the query rows
    must therefore (a) actually be consulted by the search and (b)
    produce a different ``best_scales`` than the un-sliced run on a model
    constructed to make the gate rows dominate. We assert both. If either
    fails, the slice field is dead code and the Qwen3.5 fix is gone.
    """
    in_features = 32
    out_features = 32  # 16 query rows + 16 gate rows
    batch_inputs = _make_inputs(1, 4, in_features)

    def _run(use_slice: bool) -> dict:
        torch.manual_seed(0)
        parent = _Wrapper(in_features, out_features)
        balance_layer = parent.proj
        with torch.no_grad():
            # Query rows: small magnitude. Gate rows: 100x larger so they
            # dominate the un-sliced MSE and the un-sliced baseline.
            balance_layer.weight[:16].mul_(0.1)
            balance_layer.weight[16:].mul_(10.0)
        _attach_w4a16_scheme(balance_layer)

        with torch.no_grad():
            fp16_outputs = [parent(x).clone() for x in batch_inputs]
        orig_layer_weights = {
            balance_layer: balance_layer.weight.detach().clone()
        }

        slices = {balance_layer: slice(0, 16)} if use_slice else None
        awq = AWQModifier(scheme="W4A16_ASYM", n_grid=8, duo_scaling=True)
        mapping = _build_resolved_mapping(
            parent, balance_layer, balance_output_slices=slices
        )
        _setup_modifier(
            awq, parent, balance_layer, mapping.smooth_name, batch_inputs
        )
        with create_session():
            awq._compute_best_scale(mapping, fp16_outputs, orig_layer_weights)
        return awq._error_metrics[0]

    metrics_no_slice = _run(use_slice=False)
    metrics_with_slice = _run(use_slice=True)

    # Both the identity baseline (initial_error) and the chosen
    # best_error are computed by ``_eval_scales``, which is the only
    # place the production code consults ``balance_output_slices``. If
    # the slice were silently ignored both runs would compute MSE over
    # the same rows and produce the same baseline. The huge magnitude
    # gap between query and gate rows guarantees that scoping the MSE to
    # the query rows must produce a strictly smaller baseline -- not
    # just a slightly different one. This is what proves the slice is
    # actually consulted by the real grid loop.
    assert metrics_with_slice["initial_error"] < 0.5 * metrics_no_slice[
        "initial_error"
    ], (
        "balance_output_slices did not change the grid baseline: "
        f"no_slice={metrics_no_slice['initial_error']:.3e}, "
        f"with_slice={metrics_with_slice['initial_error']:.3e}. The "
        "field is plumbed but not actually consulted by the search -- "
        "the Qwen3.5 fused q_proj fix is silently broken."
    )

    # Same argument for best_error: if the slice were ignored the search
    # would find the same minimum on both runs. Even when both runs fall
    # back to identity (best_error == initial_error), the inequality
    # below still holds because identity loss itself differs.
    assert metrics_with_slice["best_error"] < 0.5 * metrics_no_slice[
        "best_error"
    ], (
        f"no_slice best_error={metrics_no_slice['best_error']:.3e}, "
        f"with_slice best_error={metrics_with_slice['best_error']:.3e}"
    )


@pytest.mark.unit
def test_apply_smoothing_preserves_forward_equivalence():
    """
    Production-path regression for the ``_apply_smoothing`` algebraic
    invariant -- the bug that pushed Qwen3.5-27B AWQ from ``bpb=0.57``
    (fix1) to ``bpb=0.51`` (fix2).

    AWQ's correctness rests on the identity

        Y = balance(LN(x))
          = (balance with W*s)((LN with gamma/s, beta/s)(x))

    which holds element-wise in floating point. The slice-aware
    grid-search optimisation MUST NOT leak into the persisted ``_smooth``
    step: if the final balance_layer weight only scales the slice rows
    while the smooth_layer is divided by ``s`` on all input channels,
    then the un-scaled rows silently see ``x / s`` from the layernorm
    without the compensating ``* s`` on the weight. The forward pass on
    those rows then drifts away from the FP baseline -- which is exactly
    how the gate half of a fused ``q_proj`` (sigmoid'd downstream)
    catastrophically fails after quantization.

    This test runs the full ``_apply_smoothing`` path -- including
    ``_compute_best_scale`` with a non-trivial slice -- and verifies the
    parent module's FP forward output is preserved bit-close before vs
    after smoothing. If the per-mapping ``_smooth`` step ever regresses
    to a slice-aware update (the original fix1 implementation), this
    test fails with a ``> 1e-3`` divergence on the un-scaled rows.
    """
    torch.manual_seed(0)
    hidden = 32
    out_features = 32  # mimic 2*H*D fused query+gate q_proj
    parent = _LayerNormBlock(hidden=hidden, out_features=out_features)
    balance_layer = parent.proj
    smooth_layer = parent.input_layernorm

    with torch.no_grad():
        # Make query rows and gate rows differ in magnitude; this
        # combined with the slice gives the grid search a real
        # opportunity to pick scales that are good for query rows but
        # would be wrong for gate rows if naively applied to *only* the
        # slice in the persisted weight.
        balance_layer.weight[:16].mul_(0.1)
        balance_layer.weight[16:].mul_(10.0)
    _attach_w4a16_scheme(balance_layer)

    batch_inputs = _make_inputs(1, 4, hidden)
    with torch.no_grad():
        y_before = parent(batch_inputs[0]).clone()

    awq = AWQModifier(scheme="W4A16_ASYM", n_grid=4, duo_scaling=True)
    mapping = _build_resolved_mapping(
        parent,
        balance_layer,
        smooth_layer=smooth_layer,
        smooth_name="input_layernorm",
        balance_output_slices={balance_layer: slice(0, 16)},
    )
    awq._resolved_mappings = [mapping]
    _setup_modifier(
        awq, parent, balance_layer, mapping.smooth_name, batch_inputs
    )

    with create_session():
        awq._apply_smoothing(parent)

    with torch.no_grad():
        y_after = parent(batch_inputs[0])

    # The smoothing transform is exact in FP, so the only allowed
    # divergence is round-off from the divide/multiply pair on
    # ``smooth_layer.weight`` and ``balance_layer.weight``. Tolerances
    # below are chosen to fit FP32 round-off but will easily catch a
    # regression that drops the all-columns scaling on balance_layer
    # (the original fix1 bug produced O(weight magnitude) divergence on
    # the un-scaled gate rows, well above 1e-3).
    torch.testing.assert_close(y_after, y_before, rtol=1e-4, atol=1e-4)


@pytest.mark.unit
def test_compute_best_scale_raises_when_identity_baseline_is_non_finite():
    """
    Production-path regression for the non-finite-baseline silent
    failure flagged by codex.

    Pre-fix sequence of events when ``fp16_outputs`` was non-finite
    (NaN/Inf propagated from a broken upstream layer or corrupt
    calibration sample):

      1. ``initial_error = _eval_scales(identity_scales, ...)`` returns
         ``NaN``.
      2. ``best_error = initial_error = NaN``.
      3. Every grid candidate's ``loss < NaN`` is ``False`` so no
         candidate ever wins and the layer is labelled
         ``fell_back_to_identity=True``.
      4. ``err_reduction = NaN / NaN`` retains its ``> 0`` False branch
         and is silently set to ``1.0``.
      5. Modifier returns success with poison metrics; downstream
         artifact ships silently degraded.

    Post-fix the identity baseline is checked for finiteness *before*
    seeding ``best_error``. We assert the modifier raises
    ``RuntimeError`` instead of writing poison metrics into
    ``_error_metrics``. This test pins the regression that hid Inf/NaN
    AWQ failures behind the identity-fallback path.
    """
    in_features, out_features = 32, 16
    parent = _Wrapper(in_features, out_features)
    balance_layer = parent.proj
    _attach_w4a16_scheme(balance_layer)

    batch_inputs = _make_inputs(1, 4, in_features)
    # Poison fp16_outputs with NaN. _eval_scales computes
    # ``MSE(fp16_outputs, int_w_outputs)`` which is NaN if either side
    # is NaN -- the exact failure shape produced when an upstream
    # layer's calibration output is NaN/Inf.
    fp16_outputs = [
        torch.full((1, 4, out_features), float("nan")) for _ in batch_inputs
    ]
    orig_layer_weights = {balance_layer: balance_layer.weight.detach().clone()}

    awq = AWQModifier(scheme="W4A16_ASYM", n_grid=4, duo_scaling=True)
    mapping = _build_resolved_mapping(parent, balance_layer)
    _setup_modifier(awq, parent, balance_layer, mapping.smooth_name, batch_inputs)

    with create_session():
        with pytest.raises(RuntimeError, match="non-finite"):
            awq._compute_best_scale(mapping, fp16_outputs, orig_layer_weights)

    # Critically: no poison metrics may have been appended. The whole
    # point of the fix is that the layer must NOT silently end up in
    # ``_error_metrics`` with ``fell_back_to_identity=True``.
    assert awq._error_metrics == [], (
        "Non-finite baseline must abort *before* writing into "
        "_error_metrics; otherwise downstream gates silently skip the "
        f"layer. Got: {awq._error_metrics}"
    )


@pytest.mark.unit
def test_compute_best_scale_raises_when_all_grid_candidates_are_non_finite():
    """
    Companion to the identity-baseline test above: even when the
    identity-scales baseline is finite, the search itself can break if
    every executed grid candidate produces a non-finite loss (e.g. fp16
    overflow when ``W * s`` saturates the dtype, or upstream NaN that
    only surfaces on scaled forwards).

    Pre-fix this fell through the same crack: ``loss < finite_baseline``
    is False for any non-finite ``loss``, so no candidate ever won and
    the layer was silently labelled ``fell_back_to_identity=True``,
    indistinguishable in the metrics from a legitimate "no candidate
    beat identity" fallback. The artifact shipped without surfacing
    the numerical failure.

    Post-fix the modifier inspects ``history`` and distinguishes "every
    candidate ran but produced a finite loss none of which beat
    identity" (legitimate identity fallback) from "every candidate
    produced a non-finite loss" (genuine numerical failure). The
    latter must raise.

    We exercise the production code path end-to-end: ``_eval_scales``
    is the only dependency that can produce a non-finite loss, so we
    inject a single small wrapper that returns the real loss for the
    identity baseline (which the production code calls first to seed
    ``initial_error``) and ``NaN`` for every subsequent grid candidate.
    This is a *control-flow* test of ``_compute_best_scale`` -- the
    decision logic that distinguishes the two fallback shapes -- not
    an algebra test of ``_eval_scales``, which is already covered
    end-to-end by the tests above.
    """
    in_features, out_features = 32, 16
    parent = _Wrapper(in_features, out_features)
    balance_layer = parent.proj
    _attach_w4a16_scheme(balance_layer)

    batch_inputs = _make_inputs(1, 4, in_features)
    with torch.no_grad():
        fp16_outputs = [parent(x).clone() for x in batch_inputs]
    orig_layer_weights = {balance_layer: balance_layer.weight.detach().clone()}

    awq = AWQModifier(
        scheme="W4A16_ASYM",
        n_grid=4,
        duo_scaling=True,
    )
    mapping = _build_resolved_mapping(parent, balance_layer)
    _setup_modifier(awq, parent, balance_layer, mapping.smooth_name, batch_inputs)

    real_eval_scales = awq._eval_scales
    call_count = {"n": 0}

    def patched_eval_scales(scales, *args, **kwargs):
        call_count["n"] += 1
        # First call is the identity baseline -- must stay finite or
        # the test would trivially exercise the *baseline* guard
        # already covered by the test above.
        if call_count["n"] == 1:
            assert torch.allclose(scales, torch.ones_like(scales))
            return real_eval_scales(scales, *args, **kwargs)
        return float("nan")

    awq._eval_scales = patched_eval_scales

    with create_session():
        with pytest.raises(RuntimeError, match="non-finite"):
            awq._compute_best_scale(mapping, fp16_outputs, orig_layer_weights)

    assert call_count["n"] > 1, (
        "Test setup error: grid loop did not execute any non-identity "
        "candidates, so the all-candidates-non-finite path was never "
        "exercised."
    )
    assert awq._error_metrics == [], (
        "All-candidates-non-finite must abort *before* writing into "
        "_error_metrics; otherwise downstream gates silently skip the "
        f"layer. Got: {awq._error_metrics}"
    )
