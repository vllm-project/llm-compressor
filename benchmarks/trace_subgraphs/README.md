# `trace_subgraphs` Benchmark

Synthetic benchmark for measuring `trace_subgraphs` runtime at large sequential-target
counts. The default configuration exercises roughly 100K matched targets using a local
config-only `PreTrainedModel` with no Hub download.

## Usage

Run the benchmark from the repository root. Disable visible accelerators so results
measure the CPU-bound FX tracing and graph partitioning work consistently:

```bash
# Workload used for the results below
CUDA_VISIBLE_DEVICES="" python benchmarks/trace_subgraphs/benchmark.py \
    --num-targets 100000 \
    --targets-per-subgraph 300

# Quick smoke run (28 targets)
CUDA_VISIBLE_DEVICES="" python benchmarks/trace_subgraphs/benchmark.py --smoke
```

To collect three passes for a median:

```bash
for run in 1 2 3; do
    CUDA_VISIBLE_DEVICES="" python benchmarks/trace_subgraphs/benchmark.py \
        --num-targets 100000 \
        --targets-per-subgraph 300 \
        2>&1 | tee "trace_subgraphs_run_${run}.log"
done
```

Each successful run ends with a machine-readable `TRACE_BENCHMARK` line containing
the target count, subgraph count, trace time, and peak RSS.

## Results

The benchmark was run on a 12th Gen Intel Core i5-12450H under WSL2 with Python
3.13.4 and PyTorch 2.12.1 (CPU). The workload contains 100,000 matched targets and
produces 335 subgraphs with up to 300 targets each. The synthetic model uses a hidden
size of 64 and sequence length of 8, and each layer reads a shared model buffer to
produce one shared `get_attr` node. Grouping 300 targets makes the partition lists
large enough to exercise the membership-check optimization while avoiding the graph
construction overhead of creating 100,001 single-target subgraphs (`--targets-per-subgraph 1`). Results are medians
of three runs, and only `trace_subgraphs` is timed.

| Revision | Run 1 (s) | Run 2 (s) | Run 3 (s) | Median (s) |
| --- | ---: | ---: | ---: | ---: |
| Baseline (`2d7a7ea0`) | 587.682 | 574.621 | 543.853 | 574.621 |
| This PR | 37.253 | 34.790 | 33.020 | 34.790 |

Together, the changes provide a **16.5x speedup** on this workload.

The speedup comes from replacing repeated linear searches through partition node
lists with average O(1) set membership checks. Ordered node lists remain the source
of iteration order, preserving deterministic and topologically valid FX graph
construction.

The benchmark script did not exist at the baseline revision. To measure the baseline
without modifying the working tree, commit `2d7a7ea058793447faa40b75d285c7ce2111c11f`
was exported to a temporary directory and the current benchmark harness was run with
`PYTHONPATH` pointing at that export's `src/` tree. The generated `version.py` metadata
was copied into the export; it does not affect tracing behavior.

```bash
baseline=2d7a7ea058793447faa40b75d285c7ce2111c11f
baseline_dir=$(mktemp -d /tmp/llmc-baseline.XXXXXX)

git archive "$baseline" | tar -x -C "$baseline_dir"
cp src/llmcompressor/version.py "$baseline_dir/src/llmcompressor/version.py"

for run in 1 2 3; do
    CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH="$baseline_dir/src" \
    python benchmarks/trace_subgraphs/benchmark.py \
        --num-targets 100000 \
        --targets-per-subgraph 300 \
        2>&1 | tee "trace_subgraphs_baseline_run_${run}.log"
done
```

## Validation

The benchmark asserts:

- Matched target count equals `--num-targets`
- Exactly one shared `get_attr` node is present
- Subgraph count is within the expected bounds from `test_helpers.py`
- Middle subgraphs contain the expected number of sequential targets

Use `--validate-graphs` (or `--smoke`) to additionally run `graph_is_well_formed` on
every subgraph.
