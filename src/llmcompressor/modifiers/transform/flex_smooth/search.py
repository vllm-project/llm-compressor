# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Independent two-coordinate search and scale geometry for FlexSmooth."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


class NoFiniteCandidateError(ValueError):
    """All reconstruction candidates are undefined or nonfinite."""


@dataclass(frozen=True)
class SearchResult:
    alpha: float
    beta: float
    loss: float
    alpha_losses: tuple[float, ...]
    beta_losses: tuple[float, ...]


def _row_int8(value: torch.Tensor) -> torch.Tensor:
    interval = value.abs().amax(dim=-1, keepdim=True) / 127
    return (value / interval).round().clamp(-127, 127) * interval


@torch.no_grad()
def reconstruction_loss(
    activations: torch.Tensor, weights: torch.Tensor, alpha: float, beta: float
) -> float:
    """Reference-compatible INT8 proxy; intentionally not the final MXFP scheme.

    Candidate arithmetic uses storage dtype, without epsilon/clamp stabilization.
    Nonfinite candidates are retained in diagnostics and cannot win the search.
    """
    channel_scale = activations.abs().amax(0, keepdim=True).pow(alpha)
    channel_scale = channel_scale * weights.abs().amax(0, keepdim=True).pow(-beta)
    golden = activations @ weights.T
    reconstructed = (
        _row_int8(activations / channel_scale) @ _row_int8(weights * channel_scale).T
    )
    return (
        (reconstructed - golden).abs().square().mean().sqrt()
        / golden.square().mean().sqrt()
    ).item()


@torch.no_grad()
def search_alpha_beta(activations: torch.Tensor, weights: torch.Tensor) -> SearchResult:
    """Search alpha with beta=1-alpha, then beta; later equal losses win.

    A completely invalid search raises instead of treating the reference's
    default (0, 0, infinity) as a measured optimum.
    """
    if (
        activations.ndim != 2
        or weights.ndim != 2
        or activations.shape[1] != weights.shape[1]
        or not activations.numel()
        or not weights.numel()
    ):
        raise ValueError("FlexSmooth requires nonempty 2D activations and weights")
    return _search_candidates(
        lambda alpha, beta: reconstruction_loss(activations, weights, alpha, beta)
    )


def _search_candidates(evaluate: Callable[[float, float], float]) -> SearchResult:
    """Shared coordinate order and tie policy for local and reduced losses."""
    grid = tuple(round(index / 20, 2) for index in range(21))
    first = tuple(evaluate(a, 1.0 - a) for a in grid)
    best_loss, best_alpha = math.inf, 0.0
    for alpha, loss in zip(grid, first):
        if math.isfinite(loss) and loss <= best_loss:
            best_loss, best_alpha = loss, alpha
    second = tuple(evaluate(best_alpha, b) for b in grid)
    # The beta-stage threshold is the first-stage optimum, not infinity.
    best_beta = 0.0
    for beta, loss in zip(grid, second):
        if math.isfinite(loss) and loss <= best_loss:
            best_loss, best_beta = loss, beta
    if not math.isfinite(best_loss):
        raise NoFiniteCandidateError(
            "FlexSmooth search has no finite reconstruction candidate"
        )
    return SearchResult(best_alpha, best_beta, best_loss, first, second)


@torch.no_grad()
def smooth_scale(
    activation_max: torch.Tensor, weight_max: torch.Tensor, alpha: float, beta: float
) -> torch.Tensor:
    """Applied scales use the reference's FP32 clamps, unlike search candidates."""
    weight_max = weight_max.float().clamp_min(1e-5).to(weight_max.dtype)
    return (
        (activation_max.pow(alpha) / weight_max.pow(beta))
        .float()
        .clamp_min(1e-5)
        .to(activation_max.dtype)
    )


@torch.no_grad()
def ov_scales(
    activation_max: torch.Tensor,
    weight_max: torch.Tensor,
    alpha: float,
    beta: float,
    heads: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max-reduce statistics across repeated heads BEFORE computing the ratio."""
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads or weight_max.numel() % heads:
        raise ValueError("Invalid OV head geometry")
    if activation_max.shape != weight_max.shape or activation_max.ndim != 1:
        raise ValueError("OV channel statistics must be matching vectors")
    width, repeats = weight_max.numel() // heads, heads // kv_heads
    w = weight_max.float().clamp_min(1e-5).to(weight_max.dtype)
    a = activation_max.reshape(kv_heads, repeats, width).amax(1)
    w = w.reshape(kv_heads, repeats, width).amax(1)
    value_scale = smooth_scale(a, w, alpha, beta)
    output_scale = value_scale.repeat_interleave(repeats, dim=0)
    return output_scale.flatten(), value_scale.flatten()
