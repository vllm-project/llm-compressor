# Pinned Weight Staging: Why It Was Slower, and Fixing It

Status: `pinned-staging-gains` (compressed-tensors), rebased on main,
benchmarked 2026-10-30. Benchmark: `bench_pinned_staging.py` /
`run_pinned_staging.sh` (repo root), same workload as `bench_loading.py`
(Qwen/Qwen3-30B-A3B, `device_map="disk"`, perfectblend `train[:16]`,
`max_seq_length=2048`, empty recipe, weight write-back disabled to isolate
the read path).

## The problem

`DiskCache.stage(pin_memory=True)` used to stage in two steps:

1. read the tensor through safetensors into ordinary pageable memory
   (`safe_open(...).get_tensor()` — effectively an mmap view, ~17 GB/s warm)
2. `_pin_memory(staged)`: allocate a pinned buffer, then copy the whole tensor
   into it

So pinned staging moved every byte **twice** through host memory, plus one
pinned allocation per tensor (18,867 for Qwen3-30B-A3B). Meanwhile
`DiskCache.onload` moved staged tensors with a blocking `.to(device)`, which
collects none of pinned memory's async-DMA advantage.

The double copy and its memory-bandwidth contention ate most of the H2D
savings, and the unpinned onload from mmap-backed pageable memory was itself
pathological (~2 GB/s under stream contention), which is why pinning once
appeared roughly break-even.

## What main does now (compressed-tensors #937)

Main reads matching dtypes directly into a pinned buffer (one read, one
allocation, no second copy) through a raw reader that is opened per file
inside `disk_load_context` — but it validates every file with `safe_open`
before opening the raw byte stream, still serves **unpinned** reads through
`safe_open.get_tensor`, and onloads staged tensors with **synchronous**
copies.

## This branch

- **Always raw opening**: every disk read goes through `_opened_raw`. There is
  no `safe_open` in the read path at all — no validation double-open per
  shard, and no mmap-backed pageable tensors (unpinned reads land in
  allocator-owned `torch.empty` memory, which onloads ~2.7x faster, see
  below). Dtype casts read the on-disk dtype into pageable memory, cast, then
  pin; non-staged onloads read raw then transfer to the device
- **Async H2D onload**: `DiskCache.onload` moves staged pinned tensors with
  `non_blocking=True`. The copies are async DMA, stream-ordered with the
  forwards that consume them; the caching host allocator does not reuse the
  pinned blocks until the copies complete
- No kill switch: the `COMPRESSED_TENSORS_PINNED_STAGING=0` read-then-pin
  fallback is gone along with the rest of the non-raw opening logic

## Results (ps=1, mean of two warm runs per cell)

**e2e** is full process wall (interpreter start to report write, warm page
cache); **span** is the calibration window (first weight stage to last
forward); **onload** is summed H2D busy time (18,868 copies, 61.7 GB), which
partly overlaps forward:

| | e2e | span | staging wall | onload busy | forward | H2D |
|---|---|---|---|---|---|---|
| main unpinned | 125.6 s | 79.4 s | 2.2 s (mmap) | 28.7 s | 48.7 s | 2.2 GB/s |
| main pinned | 106.4 s | 61.1 s | 18.3 s (hidden) | 7.3 s | 52.6 s | 8.5 GB/s |
| branch unpinned | 112.3 s | 65.2 s | 15.8 s (hidden) | 12.6 s | 50.9 s | 4.9 GB/s |
| **branch pinned** | **105.3 s** | **59.9 s** | 18.2 s (hidden) | **4.4 s** | 53.1 s | 14.4 GB/s |

- **branch pinned is the fastest cell: -16.2% e2e / -24.6% span vs main
  unpinned**, and sits at the compute floor (53.1 s forward + the first
  subgraph's un-hidden staging)
- The async onload roughly halves main's synchronous H2D busy time (7.3 to
  4.4 s) and overlaps it with the forward pass; DMA throughput varies run to
  run (8.5-13 GB/s on main, 14.4-23.1 GB/s here) with contention from the
  forward pass
- **Even unpinned, the branch beats main by 10.6% e2e** (112.3 vs 125.6 s):
  reads into `torch.empty` pageable memory onload at ~4.9 GB/s, while the
  mmap-backed tensors `safe_open` returns collapse to ~2.2 GB/s. Staging
  gets more expensive (15.8 s of real reads vs 2.2 s of mmap), but it is
  hidden behind compute, so the span still wins
- Zero pin-copies in every cell (the read-then-pin path no longer exists);
  18,867 pinned allocations take 1.3-3.7 s of busy time on the branch
- Forward stretches by ~3-4 s under concurrent pinned staging on both main
  and the branch (memory-bandwidth contention from the staging reads)
- e2e decomposes as span + ~26 s model load (offload folder rewrite) + ~20 s
  fixed overhead (imports, tokenization, tracing). With more calibration
  samples or a real recipe, per-batch compute grows while the I/O saving
  stays fixed, so the relative e2e gain shrinks — these numbers bound it
  from above on an I/O-heavy workload

## Remaining headroom (not addressed here)

- The onload still pays ~150 us per tensor (async enqueue throttling on
  18,868 small copies). Batching H2D — staging a subgraph's tensors as views
  of shared pinned buffers and issuing one copy per buffer — would cut the
  onload toward pure DMA time (~2.5 s). That design exists on the
  `bulk-read-staging` branch
- The first big subgraph's staging has no compute to hide behind; starting it
  during data preprocessing would recover it
