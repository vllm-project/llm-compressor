r"""
Benchmark subgraph tracing with many matched sequential targets.

The dummy model contains a chain of matched target modules. Selected targets are
wrapped by modules that access registered parameters, exercising HFTracer's
parameter-name lookup path. The target count, targets per subgraph, parameter
count, and parameter-access count are configurable through the CLI.

Usage:
    python benchmarks/bench_trace_subgraphs.py \
        --num-targets 100000 \
        --targets-per-subgraph 300 \
        --num-parameters 500000 \
        --num-parameter-accesses 200 \
        --repeat 3
"""

import argparse
import gc
import math
import platform
import statistics
import time

import torch
from compressed_tensors.utils.match import match_named_modules
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs

TARGET_NAME = "DummyTarget"


class DummyConfig(PretrainedConfig):
    model_type = "trace_benchmark"

    def __init__(
        self,
        num_targets: int = 1,
        num_parameters: int = 0,
        num_parameter_accesses: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_targets = num_targets
        self.num_parameters = num_parameters
        self.num_parameter_accesses = num_parameter_accesses


class DummyTarget(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class ParameterAccessWrapper(torch.nn.Module):
    def __init__(
        self, target: DummyTarget, trace_parameter: torch.nn.Parameter
    ) -> None:
        super().__init__()
        self.target = target
        self.trace_parameter = trace_parameter

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Read the parameter here to exercise HFTracer's parameter-name lookup path.
        hidden_states = hidden_states + self.trace_parameter
        return self.target(hidden_states)


class DummySequentialModel(PreTrainedModel):
    config_class = DummyConfig

    def __init__(self, config: DummyConfig):
        super().__init__(config)
        self.lm_head = torch.nn.Linear(1, 1, bias=False)
        for index in range(config.num_parameters):
            self.register_parameter(
                f"trace_parameter_{index}",
                torch.nn.Parameter(torch.ones(1)),
            )

        target_indices = evenly_spaced_indices(
            config.num_targets, config.num_parameter_accesses
        )
        parameter_indices = evenly_spaced_indices(
            config.num_parameters, config.num_parameter_accesses
        )
        target_to_parameter_index = dict(zip(target_indices, parameter_indices))
        layers = []
        for index in range(config.num_targets):
            target = DummyTarget()
            parameter_index = target_to_parameter_index.get(index)
            if parameter_index is not None:
                target = ParameterAccessWrapper(
                    target,
                    getattr(self, f"trace_parameter_{parameter_index}"),
                )
            layers.append(target)
        self.layers = torch.nn.ModuleList(layers)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    """Return approximately evenly spaced indices in the range [0, size)."""

    if count == 0:
        return []
    if count == 1:
        return [size // 2]
    return [index * (size - 1) // (count - 1) for index in range(count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark trace_subgraphs with a dummy sequential model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        nargs="+",
        default=[100_000],
        help="One or more counts of matched sequential targets to benchmark",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of runs per target count; each run builds a fresh model",
    )
    parser.add_argument(
        "--targets-per-subgraph",
        type=int,
        default=300,
        help="Maximum number of matched targets per subgraph",
    )
    parser.add_argument(
        "--num-parameters",
        type=int,
        default=500_000,
        help="Number of synthetic torch.nn.Parameter objects to register",
    )
    parser.add_argument(
        "--num-parameter-accesses",
        type=int,
        default=200,
        help=(
            "Number of distinct registered parameters read during tracing; "
            "accesses are distributed evenly across targets and parameters"
        ),
    )
    args = parser.parse_args()

    if any(num_targets <= 0 for num_targets in args.num_targets):
        parser.error("--num-targets values must be greater than zero")
    if args.repeat <= 0:
        parser.error("--repeat must be greater than zero")
    if args.targets_per_subgraph <= 0:
        parser.error("--targets-per-subgraph must be greater than zero")
    if args.num_parameters < 0:
        parser.error("--num-parameters must be greater than or equal to zero")
    if args.num_parameter_accesses < 0:
        parser.error("--num-parameter-accesses must be greater than or equal to zero")
    if args.num_parameter_accesses > args.num_parameters:
        parser.error("--num-parameter-accesses cannot exceed --num-parameters")
    if any(
        args.num_parameter_accesses > num_targets for num_targets in args.num_targets
    ):
        parser.error("--num-parameter-accesses cannot exceed --num-targets")

    return args


def run_trace(
    num_targets: int,
    targets_per_subgraph: int,
    num_parameters: int,
    num_parameter_accesses: int,
) -> tuple[int, float]:
    # Release objects left by the previous run before building a fresh model.
    gc.collect()
    model = DummySequentialModel(
        DummyConfig(
            num_targets=num_targets,
            num_parameters=num_parameters,
            num_parameter_accesses=num_parameter_accesses,
        )
    )
    matched_targets = list(match_named_modules(model, [TARGET_NAME]))
    if len(matched_targets) != num_targets:
        raise RuntimeError(
            f"Expected {num_targets} matched targets, found {len(matched_targets)}"
        )

    sample_input = {"hidden_states": torch.zeros(1)}
    tracing_ignore = DatasetArguments().tracing_ignore

    # Collect garbage from model construction before starting the timer.
    gc.collect()
    start = time.perf_counter()
    subgraphs = trace_subgraphs(
        model=model,
        sample_input=sample_input,
        sequential_targets=[TARGET_NAME],
        ignore=tracing_ignore,
        targets_per_subgraph=targets_per_subgraph,
    )
    elapsed = time.perf_counter() - start

    # topological_partition creates one subgraph before the first target.
    expected_subgraphs = math.ceil(num_targets / targets_per_subgraph) + 1
    if len(subgraphs) != expected_subgraphs:
        raise RuntimeError(
            f"Expected {expected_subgraphs} subgraphs, found {len(subgraphs)}"
        )

    return len(subgraphs), elapsed


def main():
    args = parse_args()

    print(f"Platform: {platform.platform()}")
    print(f"Python:   {platform.python_version()}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Runs per target count: {args.repeat}")
    print(f"Targets per subgraph:  {args.targets_per_subgraph}")
    print(f"Synthetic parameters:  {args.num_parameters}")
    print(f"Parameter accesses:     {args.num_parameter_accesses}")
    print()
    print(f"{'Targets':>10} {'Subgraphs':>10} {'Median (s)':>12} {'Run times (s)'}")

    for num_targets in args.num_targets:
        timings = []
        num_subgraphs = 0
        for _ in range(args.repeat):
            num_subgraphs, elapsed = run_trace(
                num_targets=num_targets,
                targets_per_subgraph=args.targets_per_subgraph,
                num_parameters=args.num_parameters,
                num_parameter_accesses=args.num_parameter_accesses,
            )
            timings.append(elapsed)

        run_times = ", ".join(f"{elapsed:.3f}" for elapsed in timings)
        median = statistics.median(timings)
        print(
            f"{num_targets:>10,} {num_subgraphs:>10,} " f"{median:>12.3f} {run_times}"
        )


if __name__ == "__main__":
    main()
