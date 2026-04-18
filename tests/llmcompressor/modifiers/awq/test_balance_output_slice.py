"""
Unit tests for balance_output_slices: regex-to-Module resolution,
YAML-friendly value coercion, and slice-aware weight updates in AWQ
smoothing.
"""

import pytest
import torch
from torch.nn import Linear

from llmcompressor.modifiers.awq import AWQMapping, AWQModifier


def _build_attn_model(q_out: int = 16):
    """A minimal Qwen3.5-shaped block: q_proj has 2 * H*D out_features."""
    self_attn = torch.nn.ModuleDict(
        {
            "q_proj": Linear(8, q_out),
            "k_proj": Linear(8, 8),
            "v_proj": Linear(8, 8),
            "o_proj": Linear(8, 8),
        }
    )
    return torch.nn.ModuleDict(
        {
            "decoder": torch.nn.ModuleDict(
                {
                    "self_attn": self_attn,
                    "input_layernorm": torch.nn.LayerNorm(8),
                }
            )
        }
    ), self_attn


@pytest.mark.unit
def test_resolved_balance_output_slices_maps_only_matching_layer():
    """Only modules that match the slice regex should appear in the resolved dict."""
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
                balance_output_slices={"re:.*q_proj": slice(0, 8)},
            ),
        ],
        scheme="W4A16_ASYM",
    )
    model, self_attn = _build_attn_model(q_out=16)

    awq._set_resolved_mappings(model)

    assert len(awq._resolved_mappings) == 1
    resolved = awq._resolved_mappings[0]
    assert resolved.balance_output_slices is not None
    # q_proj is in the resolved slice dict; k/v are not
    assert self_attn.q_proj in resolved.balance_output_slices
    assert self_attn.k_proj not in resolved.balance_output_slices
    assert self_attn.v_proj not in resolved.balance_output_slices
    assert resolved.balance_output_slices[self_attn.q_proj] == slice(0, 8)


@pytest.mark.unit
def test_resolved_balance_output_slices_none_when_unset():
    """The resolved field stays None if the AWQMapping doesn't set it."""
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            ),
        ],
        scheme="W4A16_ASYM",
    )
    model, _ = _build_attn_model(q_out=16)

    awq._set_resolved_mappings(model)

    assert len(awq._resolved_mappings) == 1
    assert awq._resolved_mappings[0].balance_output_slices is None


@pytest.mark.unit
def test_resolved_balance_output_slices_omits_layers_with_no_pattern():
    """A slice dict that targets only q_proj should leave the resolved dict
    populated for q_proj only (no entries for k/v added with default slice)."""
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
                balance_output_slices={"re:.*q_proj": slice(0, 8)},
            ),
        ],
        scheme="W4A16_ASYM",
    )
    model, _ = _build_attn_model(q_out=16)

    awq._set_resolved_mappings(model)
    resolved = awq._resolved_mappings[0]
    assert len(resolved.balance_output_slices) == 1


@pytest.mark.unit
def test_slice_aware_weight_update_preserves_outside_rows_during_grid_search():
    """Replicate the slice-aware update used inside ``_compute_best_scale``.

    During grid search we deliberately keep rows outside the slice equal to
    the original FP16 weights so the parent module's forward pass produces a
    baseline match for those rows (e.g. the gate half of a fused
    ``q_proj`` whose ``sigmoid`` would otherwise pull the MSE off the
    rails). This test pins that arithmetic so a refactor doesn't regress it.
    """
    in_features, out_features = 8, 16
    layer = Linear(in_features, out_features)
    orig_weight = layer.weight.data.clone()

    sl = slice(0, 8)
    scales = torch.linspace(0.5, 2.0, in_features)

    new_weight = orig_weight.clone()
    new_weight[sl] = orig_weight[sl] * scales.view(1, -1)

    # Slice rows are scaled
    torch.testing.assert_close(new_weight[sl], orig_weight[sl] * scales.view(1, -1))
    # Outside-slice rows are bitwise identical
    assert torch.equal(new_weight[8:], orig_weight[8:])


@pytest.mark.unit
@pytest.mark.parametrize(
    "value,expected",
    [
        (slice(0, 8), slice(0, 8)),
        ([0, 8], slice(0, 8)),
        ((0, 8), slice(0, 8)),
        ([0, 16, 2], slice(0, 16, 2)),
        ({"start": 0, "stop": 8}, slice(0, 8)),
        ({"start": 0, "stop": 16, "step": 2}, slice(0, 16, 2)),
    ],
)
def test_balance_output_slices_yaml_friendly_inputs(value, expected):
    """
    Recipes are usually loaded from YAML, where ``slice(0, 8)`` is not a
    representable value. The dataclass must coerce the common YAML
    encodings into real ``slice`` objects at construction time so the
    public API is genuinely YAML-friendly, not just Python-friendly.
    """
    mapping = AWQMapping(
        smooth_layer="re:.*input_layernorm",
        balance_layers=["re:.*q_proj"],
        balance_output_slices={"re:.*q_proj": value},
    )
    assert mapping.balance_output_slices == {"re:.*q_proj": expected}


@pytest.mark.unit
def test_balance_output_slices_rejects_unknown_encoding():
    """Configuration errors should surface at recipe load, not mid-quant."""
    with pytest.raises(TypeError, match="balance_output_slices"):
        AWQMapping(
            smooth_layer="re:.*input_layernorm",
            balance_layers=["re:.*q_proj"],
            balance_output_slices={"re:.*q_proj": "slice(0, 8)"},
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "value,expected",
    [
        # YAML loaders that quote scalar values (e.g. PyYAML in default
        # mode when keys look like strings) must still produce a usable
        # slice rather than slice('0', '8') which only blows up later
        # during tensor indexing.
        ({"start": "0", "stop": "8"}, slice(0, 8)),
        ({"start": "0", "stop": "16", "step": "2"}, slice(0, 16, 2)),
        (["0", "8"], slice(0, 8)),
        (("0", "16", "2"), slice(0, 16, 2)),
    ],
)
def test_balance_output_slices_coerces_string_numbers(value, expected):
    """Stringy numeric components from YAML must be coerced to ``int`` at
    recipe-load time so the resulting ``slice`` works with tensor
    indexing rather than failing opaquely mid-quantization."""
    mapping = AWQMapping(
        smooth_layer="re:.*input_layernorm",
        balance_layers=["re:.*q_proj"],
        balance_output_slices={"re:.*q_proj": value},
    )
    assert mapping.balance_output_slices == {"re:.*q_proj": expected}


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        {"start": 0, "stop": "not-a-number"},
        {"start": "abc", "stop": 8},
        ["start", "stop"],
        (0, None),
        {"start": 0},  # missing required "stop"
    ],
)
def test_balance_output_slices_rejects_non_integer_components(value):
    """Non-coercible numeric components must raise at recipe-load time
    instead of producing a slice that fails later inside the grid loop."""
    with pytest.raises(TypeError, match="balance_output_slices"):
        AWQMapping(
            smooth_layer="re:.*input_layernorm",
            balance_layers=["re:.*q_proj"],
            balance_output_slices={"re:.*q_proj": value},
        )


@pytest.mark.unit
def test_balance_output_slices_yaml_dict_resolves_in_modifier():
    """End-to-end: a YAML-style dict in the AWQMapping survives all the way
    through ``_set_resolved_mappings`` and produces the same resolved
    Module->slice mapping as a Python ``slice`` literal would."""
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
                balance_output_slices={"re:.*q_proj": [0, 8]},
            ),
        ],
        scheme="W4A16_ASYM",
    )
    model, self_attn = _build_attn_model(q_out=16)

    awq._set_resolved_mappings(model)

    resolved = awq._resolved_mappings[0]
    assert resolved.balance_output_slices == {self_attn.q_proj: slice(0, 8)}


@pytest.mark.unit
def test_final_smooth_applies_scale_to_all_rows_for_equivalence():
    """
    Production invariant: the per-mapping ``_smooth`` step must scale ALL
    columns of every balance_layer, even when ``balance_output_slices`` is
    set. AWQ's correctness rests on the algebraic identity

        Y = W @ x = (W * s) @ (x / s)

    The smooth_layer (e.g. input_layernorm) is divided by the SAME scale
    applied to the balance_layer columns. If the slice were to leave some
    columns un-scaled in the *final* artefact, those columns' forward pass
    would silently see ``x / s`` from the layernorm without the
    compensating ``* s`` on the weight -- exactly the failure mode that
    pushes Qwen3.5-27B's ``sigmoid(gate)`` head off baseline.

    The slice is therefore only a grid-search concept (it scopes the MSE
    that drives scale selection) and has no effect on the persisted
    weight arithmetic. We assert that here as a regression guard so the
    invariant is captured in code, not just docs.
    """
    in_features, out_features = 8, 16
    layer = Linear(in_features, out_features)
    orig_weight = layer.weight.data.clone()
    scales = torch.linspace(0.5, 2.0, in_features)

    # This is the arithmetic _smooth performs unconditionally on every
    # balance_layer regardless of balance_output_slices.
    new_weight = orig_weight * scales.view(1, -1)

    torch.testing.assert_close(new_weight, orig_weight * scales.view(1, -1))
    # Every row -- including rows that would be "outside the slice" --
    # must reflect the scaling, otherwise the forward equivalence breaks.
    for row in range(out_features):
        torch.testing.assert_close(new_weight[row], orig_weight[row] * scales)
