# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real Gloo processes, synthetic inputs and the external ModelSlim oracle."""

import json
import os
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llmcompressor.modifiers.transform.flex_smooth.distributed import (
    search_alpha_beta_distributed,
)
from llmcompressor.modifiers.transform.flex_smooth.search import smooth_scale


def _cases():
    for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        for seed in (0, 42, 1701):
            for layout in ("uneven", "empty", "interleaved"):
                yield str(dtype), seed, layout, None
        for layout in ("uneven", "empty", "interleaved"):
            yield str(dtype), 42, layout, 5


def _inputs(dtype, seed):
    rng = torch.Generator().manual_seed(seed)
    dtype = getattr(torch, dtype.removeprefix("torch."))
    return torch.randn(17, 8, generator=rng).to(dtype), torch.randn(
        12, 8, generator=rng
    ).to(dtype)


def _worker(rank, world_size, store, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=40),
    )
    reports, errors = [], {}
    gather = dist.all_gather

    def metadata_only(outputs, value, *args, **kwargs):
        # Any attempt to gather activation/weight values fails this regression.
        assert value.dtype == torch.int64 and value.ndim == 1
        return gather(outputs, value, *args, **kwargs)

    dist.all_gather = metadata_only
    try:
        for dtype, seed, layout, budget in _cases():
            x, w = _inputs(dtype, seed)
            ids = torch.arange(x.shape[0])
            if world_size > 1:
                if layout == "uneven":
                    ids = ids[:3] if rank == 0 else ids[3:]
                elif layout == "empty":
                    ids = ids[:0] if rank == 0 else ids
                else:
                    ids = ids[rank::world_size]
            # Global IDs, not local row order, must determine a capped sample.
            ids = ids.flip(0) if budget else ids
            result = search_alpha_beta_distributed(
                x[ids], w, token_ids=ids, max_tokens=budget
            )
            scale = smooth_scale(
                result.activation_max,
                w.abs().amax(0),
                result.search.alpha,
                result.search.beta,
            )
            reports.append(
                {
                    "case": [dtype, seed, layout, budget],
                    "search": asdict(result.search),
                    "activation_max": result.activation_max.tolist(),
                    "scale": scale.tolist(),
                    "counts": result.tokens_per_rank,
                }
            )

        modes = [
            "empty",
            "nan",
            "shape",
            "budget",
            "ids",
            "duplicate",
            "zero",
            "overflow",
        ]
        if world_size > 1:
            modes.extend(
                ["duplicate_across_ranks", "weight_shape", "different_budgets"]
            )
        for mode in modes:
            x, w = _inputs("torch.float32", 42)
            ids = torch.arange(17) + rank * 17
            budget = None
            if mode == "empty":
                x = x[:0]
            elif mode == "nan" and rank == 0:
                x[0, 0] = torch.nan
            elif mode == "shape" and rank == 0:
                x = x[:, :7]
            elif mode == "budget":
                budget = 0 if rank == 0 else 5
            elif mode == "ids":
                budget = 5
                ids = None if rank == 0 else ids
            elif mode == "duplicate":
                budget = 5
                ids[:] = 0
            elif mode == "duplicate_across_ranks":
                budget = 5
                ids = torch.arange(17)
            elif mode == "weight_shape" and rank == 0:
                w = w[:3]
            elif mode == "different_budgets":
                budget = rank + 5
            elif mode == "zero":
                x.zero_()
                w.zero_()
            elif mode == "overflow":
                x.fill_(1e20)
                w.fill_(1e20)
            try:
                search_alpha_beta_distributed(x, w, token_ids=ids, max_tokens=budget)
            except ValueError as error:
                errors[mode] = str(error)
            else:
                errors[mode] = "DID NOT FAIL"
        tie = search_alpha_beta_distributed(torch.ones(3, 8), torch.ones(12, 8))
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "reports": reports,
                    "errors": errors,
                    "tie": asdict(tie.search),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        dist.all_gather = gather
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_collective_search_against_source(tmp_path, oracle, world_size):
    assert dist.is_gloo_available(), "The local DDP gate requires CPU Gloo"
    context = mp.spawn(
        _worker,
        args=(world_size, (tmp_path / "store").as_uri(), str(tmp_path)),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Gloo workers exceeded 150s; possible collective mismatch")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    ranks = [
        json.loads((tmp_path / f"rank-{i}.json").read_text()) for i in range(world_size)
    ]
    assert all(rank == ranks[0] for rank in ranks)
    assert all(value != "DID NOT FAIL" for value in ranks[0]["errors"].values())
    assert (
        ranks[0]["tie"]["alpha"],
        ranks[0]["tie"]["beta"],
        ranks[0]["tie"]["loss"],
    ) == (1, 1, 0)
    summary = []
    for actual in ranks[0]["reports"]:
        dtype, seed, layout, budget = actual["case"]
        x, w = _inputs(dtype, seed)
        selected = x if budget is None else x[:budget]
        expected = oracle.search.FlexSmoothAlphaBetaSearcher()
        alpha, beta, loss = expected.search_alpha_beta(selected, w)
        first = [
            expected.evaluate_alpha_beta(
                selected, w, round(i / 20, 2), 1 - round(i / 20, 2)
            )
            for i in range(21)
        ]
        second = [
            expected.evaluate_alpha_beta(selected, w, alpha, round(i / 20, 2))
            for i in range(21)
        ]
        got = actual["search"]
        assert (got["alpha"], got["beta"]) == (alpha, beta), actual["case"]
        tolerance = (
            1e-12
            if dtype == "torch.float64"
            else (0.02 if dtype in ("torch.bfloat16", "torch.float16") else 1e-5)
        )
        torch.testing.assert_close(
            torch.tensor(got["alpha_losses"] + got["beta_losses"], dtype=torch.float64),
            torch.tensor(first + second, dtype=torch.float64),
            rtol=tolerance,
            atol=1e-12,
        )
        assert sum(actual["counts"]) == len(selected)
        torch.testing.assert_close(
            torch.tensor(actual["activation_max"], dtype=x.dtype),
            selected.abs().amax(0),
            atol=0,
            rtol=0,
        )
        scale = oracle.scales.FlexSmoothScaleCalculator(
            alpha, beta
        ).compute_smooth_scale(
            selected.abs().amax(0), oracle.scales.compute_weight_scale(w, w.dtype)
        )
        torch.testing.assert_close(
            torch.tensor(actual["scale"], dtype=x.dtype), scale, atol=0, rtol=0
        )
        summary.append(
            {
                "case": actual["case"],
                "loss_difference": got["loss"] - loss,
                "max_candidate_difference": max(
                    abs(a - b)
                    for a, b in zip(
                        got["alpha_losses"] + got["beta_losses"], first + second
                    )
                ),
                "selection_equal": True,
                "scale_equal": True,
            }
        )
    if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        Path(output, f"collective-search-{world_size}-rank.json").write_text(
            json.dumps(
                {
                    "world_size": world_size,
                    "reference_hashes": oracle.hashes,
                    "summary": summary,
                    "ranks": ranks,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
