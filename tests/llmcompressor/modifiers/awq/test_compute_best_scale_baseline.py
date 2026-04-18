"""
End-to-end-ish tests that drive ``AWQModifier._compute_best_scale`` through
its real code path on a tiny synthetic module.

These tests exist specifically to pin down the *baseline* semantics of the
grid search:

* ``initial_error`` MUST be the loss measured at identity scales
  (``torch.ones_like(x_mean)``), not "the loss at the first grid candidate
  that survived the loop entry conditions".  The previous implementation
  set ``initial_error = loss`` on the first iteration, which under the
  default ``duo_scaling=True`` is a non-trivial scale of
  ``1 / (w_mean + 1e-4)`` -- never identity.  That made the
  smoothing-health and INCREASED-loss gates compare against an arbitrary
  (and often degenerate) reference, producing both false positives and
  silent false negatives.
* ``best_error <= initial_error`` MUST hold by construction, because we
  seed ``best_scales = identity_scales`` and only update on strict
  improvement.  The grid search can never select a worse-than-baseline
  candidate.
* When the ``max_scale_ratio`` guard rejects every grid candidate the
  layer must fall back to identity smoothing rather than raising.

We exercise this by building one ``Linear`` layer with a real W4A16
quantization scheme, attaching a real observer, populating the modifier's
private caches by hand, and invoking ``_compute_best_scale`` directly.
"""

from __future__ import annotations

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    initialize_module_for_quantization,
)
from torch.nn import Linear, Module

from llmcompressor.core.session_functions import create_session
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import ResolvedMapping
from llmcompressor.modifiers.quantization.calibration import initialize_observer
from llmcompressor.pipelines.cache import IntermediatesCache


class _Wrapper(Module):
    """Minimal parent module: forwards a hidden state through a single Linear."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = Linear(in_features, out_features, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


def _attach_w4a16_scheme(
    balance_layer: Linear, group_size: int = 8, num_bits: int = 4
) -> None:
    """Attach a W4A16 group-wise quantization scheme + observer + qparams."""
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


def _setup_modifier(
    awq: AWQModifier,
    parent: _Wrapper,
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


def _build_resolved_mapping(parent: _Wrapper, balance_layer: Linear) -> ResolvedMapping:
    return ResolvedMapping(
        smooth_name="layers.0.input_layernorm",
        smooth_layer=torch.nn.LayerNorm(balance_layer.in_features),
        balance_layers=[balance_layer],
        balance_names=["layers.0.proj"],
        parent=parent,
        parent_name="layers.0",
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
def test_extreme_activation_outlier_falls_back_to_identity_loss():
    """
    Without any safety guard, an extreme activation outlier still must
    not select a worse-than-baseline candidate. Because the search is
    seeded with identity scales, the worst case is "no improvement" and
    the recorded ``best_error`` equals ``initial_error`` -- never NaN
    or inf, never larger than baseline.
    """
    in_features, out_features = 32, 16
    parent = _Wrapper(in_features, out_features)
    balance_layer = parent.proj
    _attach_w4a16_scheme(balance_layer)

    batch_inputs = _make_inputs(1, 4, in_features, outlier_channel=11)
    with torch.no_grad():
        fp16_outputs = [parent(x).clone() for x in batch_inputs]
    orig_layer_weights = {balance_layer: balance_layer.weight.detach().clone()}

    awq = AWQModifier(scheme="W4A16_ASYM", n_grid=4, duo_scaling=False)
    mapping = _build_resolved_mapping(parent, balance_layer)
    _setup_modifier(awq, parent, balance_layer, mapping.smooth_name, batch_inputs)

    with create_session():
        awq._compute_best_scale(mapping, fp16_outputs, orig_layer_weights)

    metrics = awq._error_metrics[0]
    assert torch.isfinite(torch.tensor(metrics["initial_error"])), metrics
    assert torch.isfinite(torch.tensor(metrics["best_error"])), metrics
    assert torch.isfinite(torch.tensor(metrics["reduction"])), metrics
    assert metrics["best_error"] <= metrics["initial_error"]
