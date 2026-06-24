# AutoRound DDP Hang: Root Cause Analysis

## Symptom

AutoRound quantization hangs during `on_initialize` → `initialize_quantization` when
using `GPUS_PER_GROUP=2` (4 GPUs, 2 ranks). The same setup with `GPUS_PER_GROUP=1`
(2 GPUs, 2 ranks) completes in ~46 seconds. GPTQ does not exhibit the hang because
its examples default to `GPUS_PER_GROUP=1`.

## Root Cause: Broadcast Deadlock in `DistributedCPUCache.offload()`

### The call chain

```
initialize_quantization()
  → apply_quantization_config()
    → initialize_module_for_quantization()        # per matched Linear module
      → initialize_qparams()
        → torch.empty(shape, device=get_execution_device(module))
        → module.register_parameter(name, param)  # triggers:
          → OffloadCache.__setitem__()
            → DistributedCPUCache.offload()
              → tensor.to("cpu")                  # ⚠️ GPU→CPU copy
              → share_memory_()
              → broadcast_object_list()           # ⚠️ paired broadcast
              → barrier()                         # ⚠️ deadlock point
```

### Why it deadlocks with GPUS_PER_GROUP=2

With 4 GPUs visible (`CUDA_VISIBLE_DEVICES=0,1,2,3`), `device_map="auto_offload"`
assigns different modules to different GPUs. `get_execution_device(module)` returns
varying devices (`cuda:0`, `cuda:1`, `cuda:2`, `cuda:3`). `initialize_qparams`
creates tensors on those devices.

The `DistributedCPUCache.offload()` call chain first does a GPU→CPU copy of the
tensor. With tensors on different GPUs under different load conditions, the copy
timing varies per module. The two ranks drift out of lockstep:

- Rank 0: finishes GPU→CPU copy for module N, enters `broadcast_object_list`
- Rank 1: still doing GPU→CPU copy for module N (different GPU, different load)

`broadcast_object_list` is a paired operation — both ranks must enter it in the
same order. When timing varies, rank 0 enters broadcast N while rank 1 is still
at broadcast N-1 → **deadlock at barrier**.

The broadcasts themselves are CPU-side and fast. The GPU→CPU copy *before* each
broadcast is what desynchronizes the ranks.

### Why it works with GPUS_PER_GROUP=1

With only 2 GPUs visible (`CUDA_VISIBLE_DEVICES=1,3`), `device_map="auto_offload"`
sees limited aggregate GPU memory and assigns execution to CPU
(`onload_device=cpu`). `get_execution_device` returns `cpu` for all modules.
`initialize_qparams` creates params on CPU. `offload()` does a CPU→CPU copy —
uniform timing. The broadcasts stay paired, no deadlock.

### Why GPTQ doesn't hit this

GPTQ examples use `GPUS_PER_GROUP=1` (default). If GPTQ were run with
`GPUS_PER_GROUP=2`, it would hit the same deadlock. The hang is not specific to
AutoRound — it's a property of `DistributedCPUCache` + multi-GPU execution
devices + `initialize_quantization`.

## The Fix: `disable_onloading()` in `on_initialize`

### Mechanism

`OffloadCache` has a class-level flag `onloading_disabled`. When set:

- **`__getitem__`**: returns the offloaded (CPU) tensor directly — no CPU→GPU onload
- **`__setitem__`**: stores the value directly in `offloaded_values` — no `offload()`,
  no GPU→CPU copy, no `broadcast_object_list`, no `barrier`

This is a CT-provided escape hatch. It's already used *inside*
`initialize_module_for_quantization` (line 77 of `initialize.py`) to access
`module.weight` without triggering the distributed path.

### Implementation

```python
# llmcompressor/modifiers/autoround/base.py — on_initialize()
if QuantizationMixin.has_config(self):
    from compressed_tensors.offload import disable_onloading
    with disable_onloading():
        QuantizationMixin.initialize_quantization(self, state.model)
```

### Why this is safe

1. **Quant params are deterministic.** Both ranks compute identical scale/zero_point
   values from the same quantization scheme. No broadcast is needed — each rank
   produces the same data independently.

2. **Params stay on GPU, which is correct.** Calibration runs next — the params need
   to be on GPU for forward/backward. When the block is later offloaded to CPU, the
   params follow the normal offload path.

3. **Precedent exists.** `initialize_module_for_quantization` already uses
   `disable_onloading()` for exactly this purpose — accessing `module.weight` without
   triggering the onload path.

4. **Scoped and temporary.** The context manager restores normal behavior after
   `initialize_quantization` completes. All subsequent operations use the standard
   onload/offload path.

### Why not `force_local_cache`

`force_local_cache` only affects `cls_from_device` (new cache *creation*). During
`initialize_quantization`, the `DistributedCPUCache` instances already exist on
modules — params are added to existing caches via `__setitem__`. `force_local_cache`
has no effect on this path. The CT maintainer also rejected this approach because
it changes global cache creation semantics, which could affect model weight loading.
