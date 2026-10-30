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
  allocator-owned `torch.empty` memory, which onloads ~2.5x faster, see
  below). Dtype casts read the on-disk dtype into pageable memory, cast, then
  pin; non-staged onloads read raw then transfer to the device
- **Async H2D onload for every staged CPU tensor, pinned or pageable**:
  `DiskCache.onload` moves staged tensors with `non_blocking=True`. For
  pinned tensors the DMA engine reads host memory directly and the copies are
  stream-ordered with the forwards that consume them; for pageable tensors
  the driver stages through a pinned bounce buffer and skipping the
  post-copy synchronize still lets each copy's DMA tail overlap the next
  staging copy and the consuming kernels (~18% less H2D busy time measured)
- No kill switch: the `COMPRESSED_TENSORS_PINNED_STAGING=0` read-then-pin
  fallback is gone along with the rest of the non-raw opening logic

## Results (ps=1, mean of two warm runs per cell)

**e2e** is full process wall (interpreter start to report write, warm page
cache); **span** is the calibration window (first weight stage to last
forward); **H2D busy** is summed copy time (18,868 copies, 61.7 GB), which
partly overlaps forward:

| | e2e | span | staging wall | H2D busy | forward | H2D |
|---|---|---|---|---|---|---|
| main unpinned | 125.6 s | 79.4 s | 2.2 s (mmap) | 28.7 s | 48.7 s | 2.2 GB/s |
| main pinned | 106.4 s | 61.1 s | 18.3 s (hidden) | 7.3 s | 52.6 s | 8.5 GB/s |
| branch unpinned | 108.7 s | 62.0 s | 14.4 s (hidden) | 9.7 s | 49.8 s | 6.5 GB/s |
| **branch pinned** | **105.6 s** | **59.4 s** | 18.1 s (hidden) | **4.0 s** | 53.4 s | 15.9 GB/s |

- **branch pinned is the fastest cell: -15.9% e2e / -25.2% span vs main
  unpinned**, and sits at the compute floor (53.4 s forward + the first
  subgraph's un-hidden staging)
- The async onload roughly halves main's synchronous H2D busy time (7.3 to
  4.0 s); DMA throughput varies run to run (8.5-13 GB/s on main, 13.7-23.1
  GB/s here) with contention from the forward pass
- **Even unpinned, the branch beats main by 13.5% e2e** (108.7 vs 125.6 s)
  without allocating a single pinned byte. Of the H2D busy drop (28.7 to
  9.7 s): ~16 s comes from reading into `torch.empty` memory instead of
  `safe_open`'s mmap-backed tensors, and ~2 s from the async pageable copies
  (a controlled A/B of just the copy mode measured 12.4 to 10.2 s)
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
