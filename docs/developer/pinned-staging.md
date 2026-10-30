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

## Results (ps=1, warm CephFS page cache)

| | span | staging wall | onload | forward | H2D |
|---|---|---|---|---|---|
| main unpinned | 82.2 s | 2.1 s (mmap) | 32.1 s | 48.6 s | 1.9 GB/s |
| main pinned | 61.4 s | 19.7 s (hidden) | 4.4 s | 55.0 s | 13.0 GB/s |
| branch unpinned | 63.2 s | 16.1 s (hidden) | 12.1 s | 49.7 s | 5.2 GB/s |
| **branch pinned** | **58.0 s** | 16.6 s (hidden) | 2.8 s | 53.7 s | **23.1 GB/s** |

- **branch pinned is the fastest cell: -29.4% span vs main unpinned, -5.5%
  vs main pinned**, and sits at the compute floor (53.7 s forward + the first
  subgraph's un-hidden staging)
- The async onload more than doubles main's synchronous H2D throughput
  (23.1 vs 13.0 GB/s busy) and overlaps it with the forward pass, cutting the
  exposed onload from 4.4 s to 2.8 s
- **Even unpinned, the branch beats main by 23%** (63.2 vs 82.2 s): reads
  into `torch.empty` pageable memory onload at 5.2 GB/s, while the mmap-backed
  tensors `safe_open` returns collapse to 1.9 GB/s. Staging gets more
  expensive (16.1 s of real reads vs 2.1 s of mmap), but it is hidden behind
  compute, so the span still wins
- Zero pin-copies in every cell (the read-then-pin path no longer exists);
  18,867 pinned allocations take 1.3 s of busy time on the branch
- Forward stretches by ~4-6 s under concurrent pinned staging on both main
  and the branch (memory-bandwidth contention from the staging reads)

## Remaining headroom (not addressed here)

- The onload still pays ~150 us per tensor (async enqueue throttling on
  18,868 small copies). Batching H2D — staging a subgraph's tensors as views
  of shared pinned buffers and issuing one copy per buffer — would cut the
  onload toward pure DMA time (~2.5 s). That design exists on the
  `bulk-read-staging` branch
- The first big subgraph's staging has no compute to hide behind; starting it
  during data preprocessing would recover it
