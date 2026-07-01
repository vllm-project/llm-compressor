# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import re

import pytest
import torch

from llmcompressor.modifiers.transform.awq.mappings import (
    AWQ_MAPPING_REGISTRY,
    _exaone4_mappings,
)


@pytest.mark.unit
def test_olmo_for_causal_lm_in_registry():
    """Sanity: OlmoForCausalLM is wired to the exaone4-style mapping."""
    assert "OlmoForCausalLM" in AWQ_MAPPING_REGISTRY
    assert AWQ_MAPPING_REGISTRY["OlmoForCausalLM"] is _exaone4_mappings
    # Matches the sibling registration for Olmo3ForCausalLM.
    assert AWQ_MAPPING_REGISTRY["Olmo3ForCausalLM"] is _exaone4_mappings


@pytest.mark.unit
def test_olmo_uses_non_parametric_layernorm():
    """Validates the architectural premise for using _exaone4_mappings (which
    omits the input_layernorm / post_attention_layernorm smoothings):
    Olmo v1 / v2 use a LayerNorm with no learnable affine parameters, so a
    smoothing scale has nowhere to be absorbed at the layernorm step.
    """
    from transformers import OlmoConfig, OlmoForCausalLM

    config = OlmoConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=100,
        bos_token_id=0,
        eos_token_id=1,
    )
    with torch.device("meta"):
        model = OlmoForCausalLM(config)

    # Olmo's LayerNorm is non-parametric (elementwise_affine=False, bias=False).
    # Pull the first decoder layer's input_layernorm and confirm it has no
    # learnable affine parameter to absorb a smoothing scale into.
    input_ln = dict(model.named_modules())["model.layers.0.input_layernorm"]
    assert getattr(input_ln, "weight", None) is None, (
        "Olmo's input_layernorm must be non-parametric for _exaone4_mappings "
        "to be the correct choice (no affine to absorb a smoothing scale "
        "into); got a parametric LayerNorm instead."
    )


@pytest.mark.unit
def test_olmo_exaone4_mapping_regex_matches_real_module_tree():
    """Construct OlmoForCausalLM on the meta device with a tiny config
    and assert _exaone4_mappings' regex matches expected modules per
    layer (v_proj/o_proj attention and up_proj/down_proj MLP). No HF
    Hub downloads, no weight allocation.
    """
    from transformers import OlmoConfig, OlmoForCausalLM

    config = OlmoConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=100,
        bos_token_id=0,
        eos_token_id=1,
    )
    with torch.device("meta"):
        model = OlmoForCausalLM(config)

    module_names = [name for name, _ in model.named_modules()]

    # _exaone4_mappings entries are AWQMapping(smooth_layer, balance_layers, ...).
    # `smooth_layer` is the activation output we smooth; `balance_layers` are
    # the weight matrices that absorb the inverse scale.
    for awq_map in _exaone4_mappings:
        smooth_pat = awq_map.smooth_layer.removeprefix("re:")
        smooth_re = re.compile(smooth_pat)
        smooth_hits = [n for n in module_names if smooth_re.search(n)]
        assert smooth_hits, (
            f"OlmoForCausalLM: smooth pattern {smooth_pat!r} matched no "
            f"modules; sample names: {module_names[:20]}"
        )
        for balance_pat_raw in awq_map.balance_layers:
            balance_pat = balance_pat_raw.removeprefix("re:")
            balance_re = re.compile(balance_pat)
            balance_hits = [n for n in module_names if balance_re.search(n)]
            assert balance_hits, (
                f"OlmoForCausalLM: balance pattern {balance_pat!r} matched "
                f"no modules; sample names: {module_names[:20]}"
            )
