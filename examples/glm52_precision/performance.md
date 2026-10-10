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

## M2: optional attribution, default off

Add `--profile-attribution` to collect the JSON `attribution` object. This example
scopes temporary wrappers around existing library bindings during oneshot/save;
it changes no algorithm or offload implementation. It adds no synchronization,
tensor reads, resource queries or per-tensor logs. Wrappers preserve arguments,
return values (including async collective Work handles), exceptions and RNG.
Bindings restore when the scope exits, including on failure.

| Phase | Observed calls | Interpretation |
| --- | --- | --- |
| initialization | QuantizationModifier.on_initialize, initialize_module_for_quantization, initialize_qparams | Inclusive host wall time for configuration and qparam initialization |
| qparams / registration | initialize module's Parameter constructor, Module.register_parameter | Constructor timing excludes argument tensor allocation; initialize_qparams includes it. Registration can include nested cache work. |
| offload | CPUCache / DiskCache / distributed overrides: onload, offload, update_offload | Calls, host seconds, logical bytes, failures, grouped by active phase and implementation |
| QuaRot | Existing norm fusion consumer / rotation operation tracker | Cache counter deltas and host wall time at the existing roughly 5% progress boundaries |
| save | ModelCompressor.compress_model, to_accelerate, outer state_dict, split_torch_state_dict_into_shards, safe_save_file, from_accelerate | Actual function boundaries, including dispatch restoration after shard writes |
| synchronization | barrier, all_reduce, all_gather, broadcast, object collectives, existing torch.npu.synchronize | Host time in existing calls only; async calls stop at submission |

`operations` and `cache_by_phase_backend` use `phase/call` keys, with `calls`,
`host_seconds`, `logical_bytes` and `failures`. `quarot_segments` records completion
ranges and counter deltas. Missing version-dependent bindings appear in
`unavailable_hooks`; a missing measurement is not zero work. Model loading itself
is outside this optional scope and remains an M1 coarse stage.

All attribution times are **inclusive host wall time**, not device kernel times.
Do not sum nested qparam/registration/offload totals, base/distributed offload
implementations, or collective time into an exclusive breakdown. Distributed
offload includes both the base transfer and shared-backing collectives. Logical
bytes describe tensor payloads processed by APIs, not bytes physically read from
disk, network traffic, or storage bandwidth. Python hooks cannot see every wait
inside CT/PyTorch/HCCL.

QuaRot phase scopes follow actual tracker iteration: norm setup/reset outside
consumer iteration remains under coarse `quarot`, while consumer and rotation
work is bucketed by phase. Segment wall time can include surrounding work between
tracker resumptions; cache deltas count only calls inside the named phase. No
fusion is split, reordered or batched for instrumentation. Small models can have
more/fewer than 20 segments due to integer progress intervals.

The current save wrapper calls compress, then to_accelerate; the source performs
HF save (state_dict / reshard / shard writes) while peers wait, and finally all
ranks restore dispatch with from_accelerate. JSON separates these existing calls
and collective boundaries; no new barrier or changed write ownership is added.
The report's existing top-level save stage also covers validation and other
metadata work not separately wrapped. A shard progress bar ending does not mean
the whole save stage ended.

CPU/Gloo tiny integration exercises attribution through real initialization,
QuaRot/Flex/MXFP and compressed save/reload for CPU and disk backing. Per rank it
observed 21 module initializations and 42 qparam initializations; only the source
wrote a shard. Transformed weights, cross-rank packed state and reload checks
remain unchanged. These tiny counts do not describe the historical full model.
The NPU boundary smoke is explicitly skipped when no NPU is available.

Final local M2 validation: related L0-L3 gates 358 passed, 1 skipped (NPU),
3 known pre-existing Flex candidate-loss tolerance failures after structured
executor; generation-config tests 13 passed; MXFP tests 27 passed. Both real
two-rank CPU/Gloo and disk/Gloo tiny roundtrips passed with attribution enabled.
Ruff, formatting and diff checks passed. These are synthetic checks, not a new
real-model L4 result; the three failure tolerances and algorithm sources are
unchanged.

## M2 server acceptance and attribution questions

Use a tiny model or small existing subgraph first, not another full quantization.
Keep model/data/RNG, runtime versions, rank count and backing placement fixed;
compare attribution off/on within summary mode, then compare M1 modes.

| Hypothesis | Minimal experiment and evidence | Acceptance / decision |
| --- | --- | --- |
| Profiling preserves computation | Save transforms/scales/packed tensors, outputs, RNG state and rank-local JSON in matched tiny off/on runs | Same tensors and RNG; same operation counts/ownership, parseable reports, zero extra sync from M2 |
| Initialization is dominated by allocation, cache transfer or waits | Tiny CPU/disk-offloaded quantization initialization; collect nested qparam, registration, offload and collective counters | Hooks available; identify inclusive dominant paths before finer measurements; do not subtract overlapping totals blindly |
| Later QuaRot segments spend more time in cache APIs | Small repeated consumer/rotation workload with controlled CPU/disk backing and warm/cold repeats; collect progress deltas, external physical I/O / page-fault / resource observations | Compare seconds per consumer and per logical byte by backend; a cache-time increase alone does not prove physical disk pressure |
| Save tail is dispatch or synchronization | Tiny distributed compressed save with rank-local operation/counter JSON and timestamps | Source-only shard writes, all-rank restore dispatch, valid reload; distinguish peer barrier waits from source write time |

For a future already-planned real run, retain per-rank JSON, unfiltered logs,
versions, exact CLI/environment, cgroup/kernel incident evidence and resource
snapshots. The historical 19/49/86 minute stages cannot be retroactively split by
these new counters; rank0's missing report and SIGKILL cause remain unknown.
NPU/eight-rank overhead, output invariance and speedup are **UNVERIFIED** locally.

Disable M2 by omitting `--profile-attribution`, or revert its separate commit.
Revert M2 before M1 if removing both; neither modifies the historical checkpoint.
