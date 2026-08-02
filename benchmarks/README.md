# Benchmarks

## Subgraph tracing

[`bench_trace_subgraphs.py`](bench_trace_subgraphs.py) measures the end-to-end
runtime of `trace_subgraphs` on a synthetic sequential model, as requested in
[#2981](https://github.com/vllm-project/llm-compressor/issues/2981).

### Workload

Running the script without arguments uses this workload:

```bash
python benchmarks/bench_trace_subgraphs.py \
    --num-targets 100000 \
    --targets-per-subgraph 300 \
    --num-parameters 500000 \
    --num-parameter-accesses 200 \
    --repeat 3
```

| Setting | Value |
| --- | ---: |
| Matched sequential targets | 100,000 |
| Targets per subgraph | 300 |
| Generated subgraphs | 335 |
| Registered synthetic `Parameter` objects | 500,000 |
| Distinct `Parameter` objects accessed | 200 |
| Runs | 3 |

Each synthetic `Parameter` has one element. All 500,000 are discoverable through
`named_parameters()`, while 200 are accessed during tracing.

Each run builds a fresh model, then times only `trace_subgraphs`. Model
construction, the separate target-count validation, and pre-timing garbage
collection are excluded.

This workload triggers the existing subgraph-count warning because it groups
multiple targets per subgraph. The script independently checks for 335 subgraphs
and fails if the count differs.

### Results

Environment: Intel Core i7-13700KF, Ubuntu 26.04 under WSL2, Python 3.12.13,
PyTorch 2.13.0+cpu.

| Revision | Run 1 (s) | Run 2 (s) | Run 3 (s) | Median (s) |
| --- | ---: | ---: | ---: | ---: |
| Baseline (`28c9c76b`) | 27.085 | 26.902 | 26.455 | 26.902 |
| Optimized | 8.617 | 8.519 | 8.555 | 8.555 |

- Median speedup: **3.14x**
- Median runtime reduction: **68.2%**
