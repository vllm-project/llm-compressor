# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from llmcompressor.modifiers.transform.flex_smooth.search import (
    ov_scales,
    search_alpha_beta,
    smooth_scale,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 42, 1701])
def test_search_and_all_candidate_losses(oracle, dtype, seed):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(16, 8, generator=rng).to(dtype)
    w = torch.randn(12, 8, generator=rng).to(dtype)
    expected = oracle.search.FlexSmoothAlphaBetaSearcher()
    alpha, beta, loss = expected.search_alpha_beta(x, w)
    result = search_alpha_beta(x, w)
    assert (result.alpha, result.beta, result.loss) == (alpha, beta, loss)
    for index in range(21):
        value = round(index / 20, 2)
        assert result.alpha_losses[index] == expected.evaluate_alpha_beta(
            x, w, value, 1 - value
        )
        assert result.beta_losses[index] == expected.evaluate_alpha_beta(
            x, w, alpha, value
        )
    weight_scale = oracle.scales.compute_weight_scale(w, dtype)
    expected_scale = oracle.scales.FlexSmoothScaleCalculator(
        alpha, beta
    ).compute_smooth_scale(x.abs().amax(0), weight_scale)
    torch.testing.assert_close(
        smooth_scale(x.abs().amax(0), w.abs().amax(0), alpha, beta),
        expected_scale,
        atol=0,
        rtol=0,
    )


def test_later_ties_win_and_degenerate_search_is_explicit(oracle):
    result = search_alpha_beta(torch.ones(16, 8), torch.ones(8, 8))
    assert (result.alpha, result.beta, result.loss) == (1, 1, 0)
    with pytest.raises(ValueError, match="no finite"):
        search_alpha_beta(torch.zeros(16, 8), torch.zeros(8, 8))
    rng = torch.Generator().manual_seed(42)
    x, w = torch.randn(16, 8, generator=rng), torch.randn(8, 8, generator=rng)
    x[:, 0] = w[:, 0] = 0
    actual = search_alpha_beta(x, w)
    expected = oracle.search.FlexSmoothAlphaBetaSearcher().search_alpha_beta(x, w)
    assert (actual.alpha, actual.beta, actual.loss) == expected
    assert torch.isfinite(
        smooth_scale(x.abs().amax(0), w.abs().amax(0), actual.alpha, actual.beta)
    ).all()


@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (4, 2), (4, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_ov_scale_geometry(oracle, heads, kv_heads, dtype):
    rng = torch.Generator().manual_seed(42)
    a, w = (
        torch.rand(32, generator=rng).to(dtype),
        torch.rand(32, generator=rng).to(dtype),
    )
    a[0], w[1] = 0, 0
    clamped_w = w.float().clamp_min(1e-5).to(dtype)
    expected = oracle.scales.FlexSmoothScaleCalculator(
        0.4, 0.7, "max"
    ).compute_ov_scales(a, clamped_w, heads, kv_heads)
    actual = ov_scales(a, w, 0.4, 0.7, heads, kv_heads)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
