# L5 profiling: low-overhead observations, no numerical changes

The historical as-run source is preserved separately on
`checkpoint/production-20261008-112706`, commit `97eabf56`, with complete evidence,
patch and manifest. Do not cherry-pick its runtime mount patch into normal usage.
The production run saved 21 shards but failed post-save standalone full-dequant
reload with SIGKILL(-9). E2E did not pass; OOM is not established.

## M1: mode and time semantics

The driver accepts `--profile-mode summary|sampled|full` (default **summary**) and
`--profile-sample-every N` (default 1000, positive). Batch size, dataset, model
placement, transforms, sampling RNG and quantization recipe are unchanged.

- Summary: stage boundaries synchronize and query resources. Hot forward,
  propagation, Flex search/apply and MXFP callbacks only count/time host dispatch.
- Sampled: same host counters, with synchronization/resource queries at the first
  and each Nth hot call. Samples are stored separately, never extrapolated.
- Full: synchronous/resource-rich hot timings and original per-subgraph bars,
  intended for debugging; it deliberately perturbs asynchronous execution.

JSON schema 2 retains `stages`, `calls`, `seconds` and `cache_io`. Every stage has
`timing_kind`: `host_dispatch` or `synchronized_wall`. `host_dispatch_seconds`
includes Python, blocking calls and any existing library waits; it is **not** NPU
kernel time or CPU utilization time. `sampled_synchronized_seconds` only totals
sampled calls after draining prior device work; `sync_before_seconds` and
`sync_after_seconds` separately record host wall time spent in profiler sync.
Resource samples are boundary snapshots, not exact RSS peaks. Device allocator
peaks are process-wide. Nested durations are inclusive and must not be summed.

Summary/sampled replace only the sequential module's tqdm binding during the
example's instrumentation scope. They preserve input iteration and emit source
rank start/completed summaries at about 5% intervals, including the last subgraph.
They never filter logger sinks, warnings, errors or tracebacks. Full mode retains
the original tqdm. All bindings restore on exceptions.

## Local measurements and gates

`python -m tools.glm52_profile_benchmark --iterations 10000 --repeats 3 --output report.json`
uses fixed CPU 4x4 tensors, checks exact output/RNG equality, and reports raw
repeat timings. On the local Windows CPU runtime, median seconds were off 0.02748,
summary 0.03765, sampled 0.03891, full 0.17421. These measure small-call profiler
overhead only, **not GLM throughput or eight-rank speedup**.

Targeted mocks/CPU/Gloo tests: 48 passed. Full related L0–L3 gate: 354 passed,
3 known pre-existing Flex candidate-loss tolerance failures after structured
rotation. Failure identities are unchanged; no tolerance was relaxed.
Real NPU profiling and any production speedup remain **UNVERIFIED** locally.

## Server acceptance, without repeating full quantization

1. Use fixed tiny tensors / a small saved subgraph, fixed RNG/data/recipe and
   matched CPU/disk placement. Compare instrumentation off, summary, sampled and
   full. Check transformed weights, packed bytes, scales and outputs exactly
   (or retain the previously agreed dtype-specific numerical criterion).
2. Independently synchronize **outside** the measured experiment at its boundary.
   Measure warm/cold repeats and report host wall time, sync counts, RSS/HBM and
   I/O. Check summary has zero profiler-added hot-call synchronization/resources.
3. Check JSON labels, sample counts, rank-local counters, source progress and full
   exception tracebacks. Do not add nested timings or sum shared-page RSS by rank.
4. Only after small-run acceptance consider enabling observations on an already
   planned production task. Do not requantize the full model just for profiling.

Rollback M1 independently with `git revert <M1 commit>` on the development branch,
or select full mode temporarily for diagnostic verbosity. Neither touches the
historical checkpoint branch or server model files.
