# DDP Multi-GPU AutoRound Fixes for Large MoE Models

## Problem

Running AutoRound quantization with DDP on large MoE models (e.g., Qwen3-235B) would hang or take hours due to `DistributedCPUCache` performing a `dist.broadcast_object_list()` + `dist.barrier()` **per parameter** during offload operations (~218ms × 45K params = ~163 minutes).

## Root Cause

When `dist.is_initialized()`, `OffloadCache.cls_from_device("cpu")` returns `DistributedCPUCache` instead of `CPUCache`. This cache broadcasts every tensor to all ranks — unnecessary when each rank loads the model independently via safetensors mmap.

The bottleneck hits in two places:
1. `from_accelerate()` → `dispatch_with_map()` 
2. `set_onload_device()` in SequentialPipeline

## Fixes Applied

### Fix 1: `src/llmcompressor/utils/dev.py` — `get_main_device()` 

**Bug**: Used `rank` as the CUDA device index, which is wrong when `GPUS_PER_GROUP > 1`.  
**Fix**: Use `torch.accelerator.current_device_index()` which respects `torch.cuda.set_device()`.

```python
# Before (line 140):
return torch.device(accel_type, rank)

# After:
return torch.device(accel_type, torch.accelerator.current_device_index())
```

### Fix 2: `src/llmcompressor/modifiers/autoround/base.py` — anchor device in `apply_autoround`

**Bug**: Hardcoded `device = torch.device("cuda:0")` when `needs_multi_gpu` is true. Rank 1 with GPUs [2,3] would try to anchor on cuda:0 instead of cuda:2.  
**Fix**: Use `get_main_device()` which returns the correct per-rank device.

```python
# Before (line ~329):
device = torch.device("cuda:0")

# After:
from llmcompressor.utils.dev import get_main_device
device = get_main_device()
```

### Fix 3: `src/llmcompressor/modifiers/autoround/base.py` — GPU partition in `_update_device_map_for_dp`

**Bug**: Generated `"0,1"` for all ranks instead of per-rank GPU partitions.  
**Fix**: Offset by `local_rank * gpus_per_group`.

```python
# Before:
ar_kwargs["device_map"] = ",".join(str(i) for i in range(gpus_per_group))

# After:
local_rank = torch.distributed.get_rank()
start_gpu = local_rank * gpus_per_group
ar_kwargs["device_map"] = ",".join(str(start_gpu + i) for i in range(gpus_per_group))
```

### Patch 4 (monkey-patch, needs upstream in compressed-tensors): Force local cache

Patches `OffloadCache.cls_from_device` to return `CPUCache`/`DeviceCache` instead of `DistributedCPUCache`/`DistributedDeviceCache`. This is correct when each rank loads the model independently.

See `patch_force_local_cache()` in `test_option3_fixed.py`.

### Patch 5 (monkey-patch, needs upstream in compressed-tensors): Disable onloading during quant init

Wraps `initialize_module_for_quantization` with `disable_onloading()` to avoid per-parameter broadcast+barrier when new quantization parameters are created.

See `patch_disable_onloading_for_quant_init()` in `test_option3_fixed.py`.

## Reproduce

### Prerequisites

```bash
# Environment
source /home/yiliu7/workspace/venvs/llmc/bin/activate

# Working directory
cd /home/yiliu7/workspace/llm-compressor
```

### Run on Qwen3-8B (quick verification, ~2 minutes)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_GROUP=2 torchrun \
    --nproc_per_node=2 \
    examples/autoround/ddp/ddp_autoround.py \
    --model /storage/yiliu7/Qwen/Qwen3-8B \
    --iters 5 --nsamples 32
```

### Run on Qwen3-235B (full test, ~47 minutes)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_GROUP=2 torchrun \
    --nproc_per_node=2 \
    examples/autoround/ddp/ddp_autoround.py \
    --model /storage/yiliu7/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
    --iters 20 --nsamples 32
```

### Expected behavior

- Both ranks process all 94 layers in lockstep (~30s/layer on 235B)
- All 4 GPUs show active memory usage (~56-63 GB each)
- Each rank uses 2 GPUs: rank 0 → [0,1], rank 1 → [2,3]
- Small NCCL idle contexts (~614 MB) appear on non-owned GPUs — this is normal

### Monitor progress

```bash
# GPU utilization
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# Layer progress (from log)
grep "Applying AutoRound" /path/to/log | tail -6
```

## Known Issues

1. **8 GPU process entries in nvidia-smi**: Each of the 2 torchrun processes creates a small NCCL context (~614 MB) on all visible GPUs. Only 4 entries are doing real work (the ~56-63 GB ones). This is unavoidable without a pre-launch wrapper that restricts `CUDA_VISIBLE_DEVICES` before Python starts.

2. **OOM on layer ~11 (235B)**: With 20 iters and the full 235B model, GPU memory may be tight. Reduce `--iters` or `--nsamples` if OOM occurs.

## Upstream Plan

### PR 1: llm-compressor — Multi-GPU DDP device fixes

**Scope**: Fixes 1–3 above. Clean code changes, no monkey-patches.

**Changes**:
- `src/llmcompressor/utils/dev.py`: `get_main_device()` uses `current_device_index()` instead of `rank`
- `src/llmcompressor/modifiers/autoround/base.py`: 
  - `apply_autoround` anchor device uses `get_main_device()` instead of hardcoded `cuda:0`
  - `_update_device_map_for_dp` offsets GPU indices by `local_rank * gpus_per_group`

**Testing**: Run DDP AutoRound on Qwen3-8B with 4 GPUs (2 per rank). Verify all GPUs participate and no device mismatch errors.

---

### PR 2: compressed-tensors — Skip distributed cache when ranks have local parameters

**Problem**: `OffloadCache.cls_from_device("cpu")` unconditionally returns `DistributedCPUCache` when `dist.is_initialized()`. This causes O(n_params) broadcast+barrier ops (~218ms each) even when all ranks already have parameters locally (via independent `from_pretrained` loading with safetensors mmap).

**Proposed fix**: Add a `distributed` parameter to `cls_from_device` with auto-detection:

```python
# compressed_tensors/offload/cache/base.py

@classmethod
def cls_from_device(cls, device=None, distributed=None):
    """
    Args:
        distributed: If None (default), auto-detect based on whether
            dist is initialized. If False, always return local cache.
            If True, always return distributed cache.
    """
    if distributed is None:
        distributed = (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
    
    device_type = torch.device(device).type if device != "disk" else "disk"
    if device_type == "cpu":
        return DistributedCPUCache if distributed else CPUCache
    elif is_accelerator_type(device_type):
        return DistributedDeviceCache if distributed else DeviceCache
    elif device_type == "disk":
        return DiskCache
    ...
```

**Callers that should pass `distributed=False`**:
- `set_onload_device()` when the model was loaded independently on each rank (no meta tensors)
- Any path where the caller knows parameters are already materialized locally

**Alternative approach** — context manager:

```python
# compressed_tensors/offload/cache/base.py

_force_local_cache = threading.local()

@contextlib.contextmanager
def force_local_cache():
    """Context under which cls_from_device always returns non-distributed caches."""
    _force_local_cache.active = True
    try:
        yield
    finally:
        _force_local_cache.active = False

@classmethod
def cls_from_device(cls, device=None):
    distributed = (
        torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and not getattr(_force_local_cache, 'active', False)
    )
    ...
```

This lets llm-compressor wrap its pipeline with `force_local_cache()` without modifying every callsite.

**Testing**: 
- Existing tests pass (distributed cache still used by default)
- DDP test with independent model loading uses local cache, no broadcast overhead

---

### PR 3: compressed-tensors — Wrap quant init with `disable_onloading()`

**Problem**: `initialize_module_for_quantization` creates new parameters (scale, zero_point, etc.) which immediately trigger `DistributedCPUCache.offload()` → broadcast+barrier. These parameters are created identically on every rank, so broadcasting is always redundant.

**Proposed fix**: Wrap the function body with `disable_onloading()`:

```python
# compressed_tensors/quantization/lifecycle/initialize.py

def initialize_module_for_quantization(module, scheme=None, force_zero_point=True):
    with disable_onloading():
        # ... existing implementation ...
```

**Rationale**: New quant parameters are initialized from the quantization scheme (not from model weights), so they're identical across ranks by construction. There's no information to broadcast.

**Testing**: DDP quantization should show no broadcast calls during `initialize_module_for_quantization`. Single-process behavior unchanged.

---

### Priority

1. **PR 3** (highest): Universal fix, always correct, simple one-liner
2. **PR 2** (high): Eliminates the main bottleneck for independent-loading DDP
3. **PR 1** (medium): Required for multi-GPU-per-rank scenarios (GPUS_PER_GROUP > 1)
