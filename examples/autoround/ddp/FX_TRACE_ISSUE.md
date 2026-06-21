
# FX Trace Bottleneck in SequentialPipeline

## Problem

`trace_subgraphs()` builds an FX graph of the full model (O(n_modules)) before per-layer calibration. For 235B with 61K modules, this never finishes.

## Scope

| Modifier | Pipeline | Needs trace? | 235B hangs? |
|----------|----------|-------------|-------------|
| RTN | `DataFreePipeline` | No | Never |
| AWQ | `SequentialPipeline` | Yes | Only in DDP |
| GPTQ | `SequentialPipeline` | Yes | Only in DDP |
| AutoRound | `SequentialPipeline` | Yes | Only in DDP |

## Root cause (DDP-specific)

`load_offloaded_model()` → `from_accelerate()` → `dist.broadcast_object_list([61K-entry device_map, offload_dir])` serializes a massive dict via pickle. Rank 1's `dispatch_with_map` then creates OffloadCache for all 61K modules. Without DDP, `from_accelerate` dispatches locally — no broadcast, no wait.

## Loading strategies for 235B DDP

| Strategy | Load time | Trace | Works? |
|----------|-----------|-------|--------|
| `load_offloaded_model` + `device_map="auto"` (GPU) | 420s | Fast | No — OOM (1 GPU/rank, 178GB fills completely) |
| `load_offloaded_model` + `device_map="auto_offload"` (CPU) | 10s | Hangs | No — 61K broadcast + dispatch |
| CPU-only + sparse offload + `fast_pipeline.py` | 9s | 5s | **Yes** |

## Fixes applied

1. **`helpers.py`** — Removed `disable_onloading()` from `trace_subgraphs` (allows GPU onload)
2. **`fast_pipeline.py`** — Replaces `SequentialPipeline.__call__` with regex-based layer scanning, no FX trace. Required for 235B DDP.
3. **`distributed.py`** — Fixed `comm_device` to use `current_device()`; returns `(block, sync_fn)`
4. **`quantizer.py`** — Captures return, calls `sync_gradients()` before `_step()`
5. **`base.py`** — `_get_local_gpu_group_size()` reads `GPUS_PER_GROUP`

## Upstream plan

The FX trace is the correct architecture — it handles arbitrary model graphs. For LLMs, a fast path that regex-matches `model.layers.*` is safe. The `fast_pipeline.py` logic should move into `helpers.py` as `trace_subgraphs_fast()`, gated by a `DatasetArguments.sequential_fast_trace` flag or auto-enabled when `module_count > threshold`.

## Environment

| Component | Path |
|-----------|------|
| Python | `/home/yiliu7/workspace/venvs/llmc/bin/python` |
| torchrun | `/home/yiliu7/workspace/venvs/llmc/bin/torchrun` |
| llm-compressor | `/home/yiliu7/workspace/llm-compressor` |
| auto-round | `/home/yiliu7/workspace/ar-py` (used by venv) |
| GPUs | 8× NVIDIA B200, 180 GiB each |
| Test GPU subset | `CUDA_VISIBLE_DEVICES=0,1,2,3` |

## Required env vars

| Var | Value | Why |
|-----|-------|-----|
| `GPUS_PER_GROUP` | `2` | Triggers multi-GPU block dispatch + manual all_reduce sync |
| `AR_DISABLE_DATASET_SUBPROCESS` | `1` | Avoids `fork()` with CUDA context in `calib_dataset.py` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | GPU partition (4 GPUs for 2 ranks) |
