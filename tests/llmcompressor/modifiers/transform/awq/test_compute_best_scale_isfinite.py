"""
Regression test for the symmetric completion of the
``best_ratio == -1`` guard in ``AWQModifier._compute_best_scale``.

Post-#2640 identity is grid candidate 0, so ``initial_error`` is the
identity loss. Under IEEE 754, ``NaN < anything`` is False: if that
loss is non-finite, identity never wins, but a later finite candidate
can still flip ``best_ratio`` away from ``-1`` and bypass the original
guard. The fix raises in that case too; this file pins the contract.
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
from llmcompressor.modifiers.quantization.calibration import (
    call_observer,
    initialize_observer,
)
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.awq.mappings import ResolvedMapping
from llmcompressor.pipelines.cache import IntermediatesCache


class _Wrapper(Module):
    """Minimal parent module: forwards a hidden state through a single Linear."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = Linear(in_features, out_features, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


def _attach_w4a16_scheme(balance_layer: Linear) -> None:
    """Attach a W4A16 group-wise quantization scheme + calibrated observer."""
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            symmetric=False,
            strategy=QuantizationStrategy.GROUP,
            group_size=8,
        ),
    )
    initialize_module_for_quantization(balance_layer, scheme, force_zero_point=True)
    initialize_observer(balance_layer, base_name="weight")
    call_observer(
        balance_layer, "weight", balance_layer.weight, should_calculate_gparam=False
    )
    # Production disables this wrapper before running the fp16 baseline forward
    # in ``_apply_smoothing`` (see ``AWQModifier.on_start``); mirror that so the
    # identity-scales forward path is realistic.
    disable_quantization(balance_layer)


def _setup_modifier(
    awq: AWQModifier,
    parent: Module,
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


def _build_resolved_mapping(parent: Module, balance_layer: Linear) -> ResolvedMapping:
    return ResolvedMapping(
        smooth_name="layers.0.input_layernorm",
        smooth_layer=LayerNorm(balance_layer.in_features),
        balance_layers=[balance_layer],
        balance_names=["layers.0.proj"],
        parent=parent,
        parent_name="layers.0",
    )


@pytest.mark.unit
def test_compute_best_scale_raises_when_identity_baseline_is_non_finite():
    """
    Patch ``_compute_loss`` so only the first call (identity) returns
    NaN; the rest return the real loss. Pre-fix the run silently
    completes with NaN ``initial_error`` in ``_error_metrics``;
    post-fix it must raise. Synthetic repro of the control-flow gap,
    not a captured real-world numerical failure.
    """
    in_features, out_features = 32, 16
    parent = _Wrapper(in_features, out_features)
    balance_layer = parent.proj
    _attach_w4a16_scheme(balance_layer)

    torch.manual_seed(0)
    batch_inputs = [torch.randn(1, 4, in_features)]
    with torch.no_grad():
        fp16_outputs = [parent(x).clone() for x in batch_inputs]
    orig_layer_weights = {balance_layer: balance_layer.weight.detach().clone()}

    awq = AWQModifier(n_grid=4, duo_scaling=True)
    mapping = _build_resolved_mapping(parent, balance_layer)
    _setup_modifier(awq, parent, mapping.smooth_name, batch_inputs)

    real_compute_loss = awq._compute_loss
    call_count = {"n": 0}

    def patched_compute_loss(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return float("nan")
        return real_compute_loss(*args, **kwargs)

    awq._compute_loss = patched_compute_loss

    with create_session():
        with pytest.raises(Exception, match="No finite loss"):
            awq._compute_best_scale(mapping, fp16_outputs, orig_layer_weights)

    # Symmetric to the ``best_ratio == -1`` branch: must abort *before*
    # ``_error_metrics`` is appended, else downstream sees NaN with no signal.
    assert awq._error_metrics == [], (
        "Non-finite identity baseline must abort *before* writing into "
        f"_error_metrics; got: {awq._error_metrics}"
    )
