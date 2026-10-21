# Structured block Hadamard execution

The reported real GLM smoke spent at least 1335.41 seconds inside QuaRot before
interruption. With size 6144 and block size 32, the previous constructor expanded
192 identical signed blocks into a 6144-square matrix and every weight executed a
dense GEMM. The independent FX probe took 1.299 seconds. This change addresses
rotation execution only; it does not establish how much of the remaining QuaRot
time is norm fusion, transfers or weight writes.

## Representation and mathematical contract

`HadamardRotation` stores size, one signed block and the shifted flag. The block
size is derived from that tensor. The seeded Sylvester construction, sign placement
and normalization are unchanged. For a row vector x, let S be the signed block,
B = I ⊗ S, and P = roll(I, 16, columns):

- Ordinary: xB = reshape(reshape(x, [-1, block_size]) @ S, x.shape).
- Shifted: x(BPB) = ((xB)P)B. Multiplication by P is `roll(16)` across the
  **entire size dimension**, not separately inside each block.

The executor moves the selected axis last, selects each stride/offset segment,
performs block GEMMs in the requested FP32/FP64 precision, then restores storage
dtype once. Axis 0 still applies QᵀW; axis 1 applies WQ. Unselected elements remain
exactly unchanged. Both GEMMs are two-dimensional, avoiding matrix broadcasting.
Standard Torch operations are used, with the small block moved to the input device;
no local NPU execution is claimed.

`to_dense()` is an explicit diagnostic conversion reproducing the old B or BPB.
The Modifier never calls it. Dense Tensor input to `rotate_axis` remains supported
for diagnostic and arbitrary-matrix callers. No zero-pattern detection is used.

For n rows, ordinary rotation changes arithmetic from O(n × size²) to
O(n × size × block_size). Shifted uses two such passes and one linear roll. Matrix
storage changes from O(size²) to O(block_size²). Weight-sized intermediate tensors
still exist; this is not an offload or fusion batching change.

## CPU microbenchmark

Synthetic `[2048, 6144]` linear weights, axis 1, block 32, FP32 arithmetic, four
Torch threads, one warmup and median of five calls. Variants run in fresh processes.
Timings include `rotate_axis` allocation/copy/cast; construction is separate.
Environment: Windows, Torch 2.14.0+cpu. Raw reports are retained in the workspace
under `experiments/reports/structured-hadamard-20260928/`.

| FP32 storage | Dense | Structured | Speedup | Max abs diff | Relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 787.179 ms | 12.312 ms | 63.93× | 0 | 0 |
| Shifted | 783.674 ms | 21.697 ms | 36.12× | 4.10e-8 | 1.97e-7 |

| BF16 storage | Dense | Structured | Speedup | Max abs diff | Relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 875.854 ms | 15.740 ms | 55.65× | 0 | 0 |
| Shifted | 980.642 ms | 25.175 ms | 38.95× | 4.88e-4 | 1.32e-4 |

Rotation coefficients occupy 144 MiB dense versus 4 KiB structured. The FP32
arithmetic segment is 48 MiB in either implementation. Sampled peak RSS increases
(including construction) were 203.68 → 49.66 MiB ordinary and 587.78 → 97.89 MiB
shifted. RSS is sampled every 2 ms, includes allocator effects and can miss short
peaks. This is neither accelerator HBM usage nor an exact tensor-liveness peak.
Shifted construction took 4.540 s dense versus 0.00157 s structured.

Reproduce without a checkpoint:

```bash
python tools/benchmark_quarot_rotation.py --output benchmark-float32.json
python tools/benchmark_quarot_rotation.py --storage bfloat16 --output benchmark-bfloat16.json
```

## Correctness evidence and open gate

The new tests cover 96 dense/structured comparisons: both axes, repeated exact
strides and padded/offset segments, ordinary/shifted, three size/block combinations,
FP32/FP64 arithmetic and FP32/FP64/BF16/FP16 storage. Two additional dispatch tests
at 6144/32 forbid dense allocations and batched GEMM. Existing bias, noncontiguous,
offload, ModelSlim source and lifecycle tests remain in the gate.

Across the 96 cases, worst FP32 relative L2 is 2.90e-7, FP64 2.48e-16. BF16 storage
can round to adjacent values: worst relative L2 1.04e-3, max absolute 0.015625 on
unit-scale random inputs. Its pre-storage FP32 relative L2 stays below 1.37e-7.
FP16 storage relative L2 stays below 1.55e-4. Existing tests' numerical tolerances
are unchanged; the new tests check arithmetic and storage rounding separately.

**The existing complete-layer L4 composition gate is not yet fully passing.**
Final full gate: **319 passed, 3 failed, 0 skipped** (144.62 s). The separate MXFP
characterization gate passed **27/27**; this does not remove known ModelSlim/CT
MXFP differences. All 98 new structured-rotation checks pass. The failures are
`test_complete_layer_capture_replay_and_source_composition[0]`, `[1]`, and
`test_cli_synthetic_cannot_claim_real_l4`, all in `tests/flex_smooth/test_layer_l4.py`.
On the two synthetic complete-layer fixtures, rotation weights, activation caches,
alpha/beta, scales, final composed weights and outputs pass their original bounds.
Same-input Flex candidate losses still match the ModelSlim oracle exactly. However,
cross-path candidate losses on independently rounded post-QuaRot inputs differ by
4.73e-5 to 2.57e-4 relative L2, exceeding the existing 1e-5 criterion. Changed FP32
association is amplified by the INT8 proxy's rounding boundaries. The gate and
Flex implementation have not been relaxed or modified to hide this discrepancy.
An additional layer-0 diagnostic found one changed weight INT8 bin at each of
three alpha candidates (input norm 0.30; query norm 0.10 and 0.85), with no changed
activation bins at those candidates. The raw record is
`candidate-bin-diagnostic.json` in the evidence directory.
This remains a reported acceptance issue; local speedup alone is not approval to
claim L4 equivalence or a completed regression gate.

## Files and scope

| File | Change |
| --- | --- |
| `src/llmcompressor/modifiers/transform/quarot/rotation.py` | `HadamardRotation`, structured constructor and `rotate_axis` dispatch. |
| `src/llmcompressor/modifiers/transform/quarot/base.py` | Import and private rotation-cache type only. |
| `tests/quarot/test_structured_rotation.py` | Independent legacy dense reference, numerical comparisons and allocation/GEMM guards. |
| `tests/quarot/test_rotation.py`, `test_differential.py`, `test_topology.py` | Explicit dense materialization for matrix-only assertions; actual transforms use structured execution. |
| `tools/glm52_l4.py` | Keep dense oracle matrices for diagnostic assertions while exercising structured weight rotation. |
| `tools/benchmark_quarot_rotation.py` | Reproducible CPU timing, sampled memory and accuracy measurements. |

Norm fusion, expert batching, distributed write protocol, SequentialPipeline,
FlexSmooth, MXFP policy, recipe and save/reload implementations are unchanged.
An eventual A5 re-profile should use the same original BF16 checkpoint, recipe,
sample count and sequence length, with a fresh output directory. Return the log and
profile JSON, especially QuaRot time, cache calls/bytes and CPU/NPU memory. The
hypothesis is less rotation compute and smaller matrix transfers; end-to-end speedup
and any remaining fusion/I/O bottleneck must be measured on that server.
