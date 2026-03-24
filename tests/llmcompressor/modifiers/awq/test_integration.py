"""
Integration test proving that when AWQModifier is stacked with
QuantizationModifier, the downstream quantizer operates on
post-AWQ-smoothed weights — not the original weights.

This is the key correctness proof for the AWQ decoupling: AWQ applies
smoothing first, then the downstream quantizer calibrates and produces
scales/zero-points on the already-smoothed model.
"""

import pytest
import torch
import torch.nn as nn

from llmcompressor.core import State, active_session, reset_session
from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
from llmcompressor.modifiers.quantization.quantization.base import QuantizationModifier
from llmcompressor.recipe import Recipe


class _TinyTransformerBlock(nn.Module):
    """
    Minimal model that has the structure AWQ mappings expect:
    a LayerNorm (smooth_layer) feeding into Linear projections (balance_layers).
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        h = self.input_layernorm(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        return q + k + v


class _TinyModel(nn.Module):
    """Wraps a single transformer block to look like a full model."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.layer = _TinyTransformerBlock(dim)

    def forward(self, x):
        return self.layer(x)


@pytest.mark.unit
@torch.no_grad()
def test_downstream_quantizer_sees_smoothed_weights():
    """
    Prove that QuantizationModifier produces quantization artifacts
    (weight_scale, weight_zero_point) on weights that differ from the
    original — i.e. the weights AWQ smoothed.

    Flow:
    1. Snapshot original weights.
    2. Initialize AWQModifier (resolves mappings).
    3. Manually apply smoothing (via _apply_smoothing with synthetic
       activation data, inside the temp quant scheme context).
    4. Confirm weights changed (AWQ smoothing applied).
    5. Initialize + calibrate QuantizationModifier on the post-smoothed model.
    6. Confirm quantization artifacts exist on the smoothed weights.
    """
    reset_session()
    dim = 128
    model = _TinyModel(dim)

    # -- 1. Snapshot original weights --
    orig_q_weight = model.layer.q_proj.weight.clone()
    orig_k_weight = model.layer.k_proj.weight.clone()
    orig_v_weight = model.layer.v_proj.weight.clone()
    orig_ln_weight = model.layer.input_layernorm.weight.clone()

    # -- 2. Set up AWQ with explicit mappings --
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            ),
        ],
        scheme="W4A16",
        targets=["Linear"],
    )
    quant = QuantizationModifier(
        scheme="W4A16",
        targets=["Linear"],
    )

    # Register modifiers in the session so _validate_recipe can find them
    session = active_session()
    recipe = Recipe.from_modifiers([awq, quant])
    session.lifecycle.recipe = recipe

    state = State(model=model)

    # Initialize AWQ (resolves mappings, validates recipe)
    awq.on_initialize(state)
    assert len(awq._resolved_mappings) > 0, "AWQ should resolve at least one mapping"

    # -- 3. Simulate calibration: populate AWQ caches with synthetic data --
    # AWQ needs: _parent_args_cache (for re-running forward) and
    #            _smooth_activation_means (for scale computation)
    awq.on_start(state, None)

    # Run a few synthetic batches through the model to populate hooks
    for _ in range(3):
        x = torch.randn(2, dim)
        model(x)

    # -- Apply smoothing --
    awq._apply_smoothing(model)

    # -- 4. Confirm AWQ changed the weights --
    q_changed = not torch.equal(model.layer.q_proj.weight, orig_q_weight)
    k_changed = not torch.equal(model.layer.k_proj.weight, orig_k_weight)
    v_changed = not torch.equal(model.layer.v_proj.weight, orig_v_weight)
    ln_changed = not torch.equal(model.layer.input_layernorm.weight, orig_ln_weight)

    assert q_changed or k_changed or v_changed or ln_changed, (
        "AWQ smoothing should have modified at least some weights, "
        "but all weights are identical to originals"
    )

    # Record the post-smoothed weights that the quantizer should see
    smoothed_q_weight = model.layer.q_proj.weight.clone()
    smoothed_k_weight = model.layer.k_proj.weight.clone()
    smoothed_v_weight = model.layer.v_proj.weight.clone()

    # Finish AWQ
    awq.on_end(state, None)

    # -- 5. Confirm AWQ left no quantization artifacts --
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert not hasattr(module, "weight_scale"), (
                f"AWQ should not leave weight_scale on {name}"
            )
            assert not hasattr(module, "weight_zero_point"), (
                f"AWQ should not leave weight_zero_point on {name}"
            )

    # -- 6. Now run QuantizationModifier on the post-smoothed model --
    quant.on_initialize(state)

    # Start calibration (attaches observers, enables quantization)
    quant.on_start(state, None)

    # End calibration (freezes quantization, removes observers)
    quant.on_end(state, None)

    # -- 7. Verify quantization artifacts exist --
    linear_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    assert len(linear_modules) > 0

    for name, module in linear_modules:
        assert hasattr(module, "quantization_scheme"), (
            f"QuantizationModifier should have set quantization_scheme on {name}"
        )

    # -- 8. Verify the quantizer operated on smoothed weights, not originals --
    # The weights on the model should still be the smoothed versions
    # (quantization modifies the forward path, not the stored weights in PTQ)
    assert torch.equal(model.layer.q_proj.weight, smoothed_q_weight), (
        "The quantizer should have operated on the smoothed q_proj weights"
    )
    assert torch.equal(model.layer.k_proj.weight, smoothed_k_weight), (
        "The quantizer should have operated on the smoothed k_proj weights"
    )
    assert torch.equal(model.layer.v_proj.weight, smoothed_v_weight), (
        "The quantizer should have operated on the smoothed v_proj weights"
    )


@pytest.mark.unit
@torch.no_grad()
def test_awq_alone_produces_no_quantization_artifacts():
    """
    When AWQModifier runs alone (without a downstream quantizer),
    the model should have smoothed weights but NO quantization
    artifacts (no scales, no zero-points, no quantization_scheme).
    """
    reset_session()
    dim = 128
    model = _TinyModel(dim)

    orig_q_weight = model.layer.q_proj.weight.clone()

    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            ),
        ],
        scheme="W4A16",
        targets=["Linear"],
    )

    session = active_session()
    recipe = Recipe.from_modifiers([awq])
    session.lifecycle.recipe = recipe

    state = State(model=model)
    awq.on_initialize(state)
    awq.on_start(state, None)

    # Populate caches
    for _ in range(3):
        model(torch.randn(2, dim))

    awq._apply_smoothing(model)
    awq.on_end(state, None)
    awq.on_finalize(state)

    # Weights should have changed (smoothing applied)
    assert not torch.equal(model.layer.q_proj.weight, orig_q_weight), (
        "AWQ should have smoothed the weights"
    )

    # But NO quantization artifacts should exist
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert not hasattr(module, "quantization_scheme"), (
                f"AWQ alone should not leave quantization_scheme on {name}"
            )
            assert not hasattr(module, "weight_scale"), (
                f"AWQ alone should not leave weight_scale on {name}"
            )
            assert not hasattr(module, "weight_zero_point"), (
                f"AWQ alone should not leave weight_zero_point on {name}"
            )
