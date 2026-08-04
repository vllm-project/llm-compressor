"""
Benchmark: trace_subgraphs at ~100K sequential targets.

Usage:
    # full scale (CPU, no Hub download)
    python benchmarks/trace_subgraphs/benchmark.py --num-targets 100000

    # quick smoke (~28 layers, mirrors test_helpers scale)
    python benchmarks/trace_subgraphs/benchmark.py --smoke

    # fewer subgraphs (same targets, higher targets_per_subgraph)
    python benchmarks/trace_subgraphs/benchmark.py \\
        --num-targets 100000 --targets-per-subgraph 4
"""

from __future__ import annotations

import argparse
import math
import platform
import resource
import sys
import time

import torch
import torch.nn as nn
from compressed_tensors.utils.match import match_named_modules
from loguru import logger
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from llmcompressor.args import DatasetArguments
from llmcompressor.pipelines.sequential.helpers import (
    graph_is_well_formed,
    trace_subgraphs,
)
from llmcompressor.utils.dev import skip_weights_initialize

SEQUENTIAL_TARGET = "TraceTargetLayer"
DEFAULT_NUM_TARGETS = 100_000
SMOKE_NUM_TARGETS = 28


class TraceTargetConfig(PretrainedConfig):
    model_type = "trace_target_benchmark"

    def __init__(
        self,
        hidden_size: int = 64,
        num_hidden_layers: int = DEFAULT_NUM_TARGETS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class TraceTargetLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


class TraceTargetModel(PreTrainedModel):
    config_class = TraceTargetConfig

    def __init__(self, config: TraceTargetConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            TraceTargetLayer(config.hidden_size)
            for _ in range(config.num_hidden_layers)
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trace_subgraphs")
    parser.add_argument(
        "--num-targets",
        type=int,
        default=DEFAULT_NUM_TARGETS,
        help="Number of matched sequential targets (layers)",
    )
    parser.add_argument(
        "--targets-per-subgraph",
        type=int,
        default=1,
        help="Number of sequential targets grouped per subgraph",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size for the synthetic model",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Sequence length for sample hidden states",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Run a small smoke benchmark with {SMOKE_NUM_TARGETS} targets",
    )
    parser.add_argument(
        "--validate-graphs",
        action="store_true",
        help="Run graph_is_well_formed on every subgraph (slow at large scale)",
    )
    return parser.parse_args()


def peak_rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024**3)
    return usage / (1024**2)


def build_model(num_targets: int, hidden_size: int) -> TraceTargetModel:
    config = TraceTargetConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_targets,
    )
    with skip_weights_initialize():
        model = TraceTargetModel(config)
    model.eval()
    return model


def validate_subgraphs(
    model: TraceTargetModel,
    subgraphs: list,
    num_targets: int,
    targets_per_subgraph: int,
    validate_graphs: bool,
) -> None:
    targets = {module for _, module in match_named_modules(model, [SEQUENTIAL_TARGET])}
    assert len(targets) == num_targets, (
        f"matched {len(targets)} targets, expected {num_targets}"
    )

    min_subgraphs = num_targets // targets_per_subgraph + 1
    max_subgraphs = math.ceil(num_targets / targets_per_subgraph) + 1
    assert min_subgraphs <= len(subgraphs) <= max_subgraphs, (
        f"expected [{min_subgraphs}, {max_subgraphs}] subgraphs, got {len(subgraphs)}"
    )

    for subgraph in subgraphs[1:-1]:
        num_targets_present = sum(
            1
            for module in subgraph.submodules(model)
            if module.__class__.__name__ == SEQUENTIAL_TARGET
        )
        assert num_targets_present == targets_per_subgraph

    if validate_graphs:
        for subgraph in subgraphs:
            assert graph_is_well_formed(subgraph.graph)


def main() -> None:
    args = parse_args()
    num_targets = SMOKE_NUM_TARGETS if args.smoke else args.num_targets

    logger.info("=" * 60)
    logger.info("BENCHMARK — trace_subgraphs")
    logger.info("=" * 60)
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Targets: {num_targets:,}")
    logger.info(f"Targets/subgraph: {args.targets_per_subgraph}")
    logger.info(f"Hidden size: {args.hidden_size}")
    logger.info(f"Seq len: {args.seq_len}")

    build_start = time.perf_counter()
    model = build_model(num_targets, args.hidden_size)
    build_s = time.perf_counter() - build_start

    sample_input = {
        "hidden_states": torch.zeros(1, args.seq_len, args.hidden_size),
    }
    ignore = DatasetArguments().tracing_ignore

    match_start = time.perf_counter()
    matched_targets = {
        module for _, module in match_named_modules(model, [SEQUENTIAL_TARGET])
    }
    match_s = time.perf_counter() - match_start

    trace_start = time.perf_counter()
    subgraphs = trace_subgraphs(
        model,
        sample_input,
        sequential_targets=[SEQUENTIAL_TARGET],
        ignore=ignore,
        targets_per_subgraph=args.targets_per_subgraph,
    )
    trace_s = time.perf_counter() - trace_start

    validate_subgraphs(
        model,
        subgraphs,
        num_targets,
        args.targets_per_subgraph,
        validate_graphs=args.smoke or args.validate_graphs,
    )

    rss_gb = peak_rss_gb()
    targets_per_second = num_targets / trace_s if trace_s > 0 else float("inf")

    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS — trace_subgraphs")
    logger.info("=" * 60)
    logger.info(f"Sequential target: {SEQUENTIAL_TARGET}")
    logger.info(f"Matched targets: {len(matched_targets):,}")
    logger.info(f"Subgraphs: {len(subgraphs):,}")
    logger.info(f"Build time: {build_s:.1f}s")
    logger.info(f"Match time: {match_s:.1f}s")
    logger.info(f"Trace time: {trace_s:.1f}s ({trace_s / 60:.2f} min)")
    logger.info(f"Throughput: {targets_per_second:,.0f} targets/s")
    logger.info(f"Peak RSS: {rss_gb:.2f} GB")
    logger.info("=" * 60)

    print(
        "TRACE_BENCHMARK "
        f"targets={num_targets} "
        f"subgraphs={len(subgraphs)} "
        f"trace_s={trace_s:.3f} "
        f"peak_rss_gb={rss_gb:.2f}"
    )


if __name__ == "__main__":
    main()
