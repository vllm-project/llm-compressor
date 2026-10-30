# Pinned Weight Staging: Why It Was Slower, and Fixing It

Status: implemented on `main` (compressed-tensors), benchmarked 2026-10-30.
Benchmark: `bench_pinned_staging.py` / `run_pinned_staging.sh` (repo root), same
workload as `bench_loading.py` (Qwen/Qwen3-30B-A3B, `device_map="disk"`,
perfectblend `train[:16]`, `max_seq_length=2048`, empty recipe, weight
write-back disabled to isolate the read path).

## The problem

`DiskCache.stage(pin_memory=True)` on main staged in two steps:

1. read the tensor through safetensors into ordinary pageable memory
   (`safe_open(...).get_tensor()` — effectively an mmap view, ~17 GB/s warm)
2. `_pin_memory(staged)`: allocate a pinned buffer, then copy the whole tensor
   into it

So pinned staging moved every byte **twice** through host memory, plus one
pinned allocation per tensor (18,867 for Qwen3-30B-A3B). Meanwhile
`DiskCache.onload` moved staged tensors with a blocking `.to(device)`, which
collects none of pinned memory's async-DMA advantage.

Measured effect (ps=1 / ps=6, warm CephFS page cache):

| | span | staging wall | onload | forward | note |
|---|---|---|---|---|---|
| unpinned | 83.8 / 78.3 s | 2.1 / 2.1 s | 29.7 / 28.4 s | 52.5 / 48.9 s | mmap staging nearly free; **onload from pageable memory collapses to ~2.2 GB/s** under stream contention (per-copy sync on 18,870 blocking copies) |
| pinned (old) | 70.6 / 74.6 s | 41.6 / 44.0 s | 7.8 / 5.2 s | 60.5 / 61.0 s | 35.7 / 38.5 s of pin-copy; the extra memory traffic **stretches forward by 8-12 s**; at ps=6 the staging wall also exceeds the compute window (8.4 s exposed) |

This is why pinning appeared worse than generic CPU memory: the double copy
and its memory-bandwidth contention ate most of the H2D savings, and at larger
subgraphs the staging itself became exposed. (On this machine the unpinned
onload is so pathological that old-pinned still came out slightly ahead; with a
less catastrophic pageable H2D path, old-pinned loses outright.)

## The fix (compressed-tensors)

- `disk_utils._opened_raw`: a cached, header-parsed, unbuffered raw reader
  alongside the existing `safe_open` handle cache (thread-local, bounded,
  evicted before in-place rewrites in `update_offload`)
- `load_disk_tensor_from_offload(..., pin_memory=True)` reads matching dtypes
  **directly into a pinned buffer** — one read, one allocation, no
  intermediate pageable tensor, no second copy. Dtype casts and big-endian
  hosts fall back to the safetensors path (cast, then pin), preserving the
  existing conversion contract
- `DiskCache.stage` routes `pin_memory` through the loader instead of
  pinning after the fact
- `DiskCache.onload` moves staged pinned tensors with
  `non_blocking=True`: the H2D copies are async DMA, stream-ordered with the
  forwards that consume them; the caching host allocator does not reuse the
  pinned blocks until the copies complete
- Kill switch: `COMPRESSED_TENSORS_PINNED_STAGING=0` restores main's exact
  read-then-pin behavior (used for the "pinned (old)" benchmark column)

## Results

| | span (ps=1) | span (ps=6) | staging wall | onload | forward | H2D |
|---|---|---|---|---|---|---|
| unpinned | 83.8 s | 78.3 s | 2.1 s | 29.7 / 28.4 s | 52.5 / 48.9 s | 2.1-2.2 GB/s |
| pinned (old) | 70.6 s | 74.6 s | 41.6 / 44.0 s | 7.8 / 5.2 s | 60.5 / 61.0 s | 7.5-10.5 GB/s |
| **pinned (new)** | **59.3 s** | **59.4 s** | 18.9 / 17.8 s (hidden) | 5.5 / 2.7 s | 52.2 / 51.6 s | 10.1-18.4 GB/s |

- **+29.2% / +24.2% span reduction vs unpinned**, +15.7% / +18.9% vs the old
  pinned path, at `sequential_targets_per_subgraph` 1 and 6
- Zero pin-copies (0 vs 35.7-38.5 s of extra memory traffic); forward time
  matches the unpinned baseline (no contention stretch)
- The span now sits at the compute floor: forward + a 2.7-5.5 s onload + the
  first subgraph's un-hidden staging

## Remaining headroom (not addressed here)

- The onload still pays ~160-330 us per tensor (async enqueue throttling on
  18,870 small copies). Batching H2D — staging a subgraph's tensors as views
  of shared pinned buffers and issuing one copy per buffer — would cut the
  onload toward pure DMA time (~2.5 s). That design exists on the
  `bulk-read-staging` branch
- The first big subgraph's staging has no compute to hide behind (~2-9 s
  exposed at ps>=6); starting it during data preprocessing would recover it
