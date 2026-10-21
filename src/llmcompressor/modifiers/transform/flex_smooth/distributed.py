# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Collective search used by the distributed FlexSmooth Modifier.

All group members must call in the same order with replicated weights from the
same snapshot. Only metadata, maxima and scalar moments are communicated. CPU
Gloo is tested; this module does not initialize a backend. The Modifier currently
uses the uncapped path; global token-ID selection remains an independent primitive.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .search import SearchResult, _row_int8, _search_candidates


@dataclass(frozen=True)
class CollectiveSearchResult:
    search: SearchResult
    activation_max: torch.Tensor
    tokens_per_rank: tuple[int, ...]


def _gather(value, group):
    copies = [torch.empty_like(value) for _ in range(dist.get_world_size(group))]
    dist.all_gather(copies, value, group=group)
    return copies


@torch.no_grad()
def search_alpha_beta_distributed(
    activations: torch.Tensor,
    weights: torch.Tensor,
    *,
    token_ids: torch.Tensor | None = None,
    max_tokens: int | None = None,
    group=None,
) -> CollectiveSearchResult:
    """Search one global token set with MAX statistics and SUM loss moments.

    Empty local inputs are allowed; an empty global set is not. A budget selects
    the smallest unique global token IDs, independent of rank assignment. IDs
    must encode original sample/token order, not rank-local batch indices. Each
    rank need retain only its smallest N IDs/rows for a global budget N. No raw
    activations are gathered. Duplicate candidate IDs are rejected (including
    DistributedSampler padding); callers must supply a deduplicated token set.

    INT8 proxy GEMMs, subtraction, square, sqrt and division retain storage dtype.
    Mean moments are accumulated/reduced in FP32 (FP64 for FP64 inputs), then
    cast back before sqrt. This preserves the reference's square overflow and
    nonfinite-candidate policy, but reduction/GEMM ordering is not bit-exact to
    its single-process mean. Weights must be identical on all ranks: shape and
    dtype are checked here, replica contents belong to the caller's contract.
    """
    if not dist.is_initialized():
        raise ValueError("Distributed FlexSmooth search requires a process group")
    dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    budget_ok = max_tokens is None or (
        isinstance(max_tokens, int)
        and not isinstance(max_tokens, bool)
        and 0 < max_tokens < torch.iinfo(torch.int64).max
    )
    valid = (
        activations.ndim == weights.ndim == 2
        and weights.numel() > 0
        and activations.shape[1] == weights.shape[1]
        and activations.dtype == weights.dtype
        and weights.dtype in dtypes
        and activations.device == weights.device
        and activations.layout == weights.layout == torch.strided
        and budget_ok
    )
    if valid and max_tokens is not None:
        valid = (
            token_ids is not None
            and token_ids.dtype == torch.int64
            and token_ids.device == activations.device
            and token_ids.shape == (activations.shape[0],)
            and bool((token_ids >= 0).all())
            and bool((token_ids < torch.iinfo(torch.int64).max).all())
            and token_ids.unique().numel() == token_ids.numel()
        )
    if valid:
        valid = bool(
            torch.isfinite(activations).all() and torch.isfinite(weights).all()
        )
    # Fixed-size handshake keeps malformed/empty ranks on the same error path.
    metadata = torch.tensor(
        [
            int(valid),
            weights.shape[0] if weights.ndim == 2 else -1,
            weights.shape[1] if weights.ndim == 2 else -1,
            dtypes.index(weights.dtype) if weights.dtype in dtypes else -1,
            max_tokens if budget_ok and max_tokens is not None else -1,
        ],
        device=weights.device,
        dtype=torch.int64,
    )
    metadata = [value.tolist() for value in _gather(metadata, group)]
    if not all(value[0] for value in metadata) or any(
        value[1:] != metadata[0][1:] for value in metadata
    ):
        raise ValueError("Invalid or inconsistent distributed FlexSmooth inputs")

    if max_tokens is not None:
        indices = token_ids.argsort()[:max_tokens]
        local_ids = token_ids[indices]
        length = torch.tensor(
            [local_ids.numel()], device=weights.device, dtype=torch.int64
        )
        width = max(int(value.item()) for value in _gather(length, group))
        if not width:
            raise ValueError(
                "Distributed FlexSmooth requires global calibration tokens"
            )
        sentinel = torch.iinfo(torch.int64).max
        packed = torch.full(
            (width,), sentinel, device=weights.device, dtype=torch.int64
        )
        packed[: local_ids.numel()] = local_ids
        ids = torch.cat(_gather(packed, group))
        ids = ids[ids != sentinel]
        if ids.unique().numel() != ids.numel():
            raise ValueError("Duplicate global FlexSmooth token IDs")
        cutoff = ids.sort().values[:max_tokens][-1]
        activations = activations[indices[local_ids <= cutoff]]

    count = torch.tensor(
        [activations.shape[0]], device=weights.device, dtype=torch.int64
    )
    counts = tuple(int(value.item()) for value in _gather(count, group))
    if not sum(counts):
        raise ValueError("Distributed FlexSmooth requires global calibration tokens")
    accumulation_dtype = (
        torch.float64 if weights.dtype == torch.float64 else torch.float32
    )
    amax = (
        activations.abs().amax(0).to(accumulation_dtype)
        if activations.shape[0]
        else torch.zeros(
            weights.shape[1], device=weights.device, dtype=accumulation_dtype
        )
    )
    dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=group)
    amax = amax.to(weights.dtype)
    wmax = weights.abs().amax(0)
    golden = activations @ weights.T
    numel = sum(counts) * weights.shape[0]

    def evaluate(alpha, beta):
        moments = torch.zeros(2, device=weights.device, dtype=accumulation_dtype)
        if activations.shape[0]:
            scale = amax.pow(alpha) * wmax.pow(-beta)
            reconstructed = (
                _row_int8(activations / scale) @ _row_int8(weights * scale).T
            )
            # Divide before summing to avoid overflow of an otherwise finite mean.
            moments[0] = (
                (reconstructed - golden).abs().square().to(accumulation_dtype) / numel
            ).sum()
            moments[1] = (golden.square().to(accumulation_dtype) / numel).sum()
        dist.all_reduce(moments, op=dist.ReduceOp.SUM, group=group)
        means = moments.to(weights.dtype)
        return (means[0].sqrt() / means[1].sqrt()).item()

    return CollectiveSearchResult(_search_candidates(evaluate), amax, counts)
