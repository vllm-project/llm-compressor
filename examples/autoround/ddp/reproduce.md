# Multi-GPU DDP AutoRound Reproduce

## Command

```bash
cd /home/yiliu7/workspace/llm-compressor

AR_DISABLE_DATASET_SUBPROCESS=1 CUDA_VISIBLE_DEVICES=0,1,6,7 GPUS_PER_GROUP=2 NPROC=2 MASTER_PORT=29501 \
  bash examples/autoround/ddp/launch_multi_gpu.sh \
  ddp_qwen3_multi_gpu_example.py \
  --model /storage/yiliu7/Qwen/Qwen3-8B \
  --gpus-per-group 2 \
  --scheme W4A16 \
  --nsamples 32 --iters 50 \
  > /tmp/multi_gpu_test.log 2>&1 &
```

## Monitor

```bash
# Check progress
tail -f /tmp/multi_gpu_test.log
# Check processes
ps aux | grep ddp_qwen3_multi | grep -v grep
# Check GPU usage
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# Kill
pkill -f ddp_qwen3_multi_gpu_example
```

## Current State

- ✅ 4 code changes implemented (launch_multi_gpu.sh, base.py, distributed.py, quantizer.py)
- ✅ Model loading works with `device_map="auto"` (dispatch 547/547 in <1s)
- ✅ GPU partitioning works (rank 0 → GPUs 0,1; rank 1 → GPUs 2,3)
- 🔄 **Hang** after "Disabling tokenizer parallelism" warning — inside `get_dataset()`
  - `AR_DISABLE_DATASET_SUBPROCESS=1` avoids the fork issue
  - Dataset is cached, not downloading
  - Both processes at ~100% CPU but no progress

## Key Files

| File | Change |
|------|--------|
| `examples/autoround/ddp/ddp_qwen3_multi_gpu_example.py` | NEW — multi-GPU DDP example |
| `examples/autoround/ddp/launch_multi_gpu.sh` | NEW — bash wrapper for GPU partitioning |
| `src/llmcompressor/modifiers/autoround/base.py` | `_update_device_map_for_dp` + auto_offload gate use `GPUS_PER_GROUP` |
| `auto_round/utils/distributed.py` | `setup_ddp_if_needed_` returns `(block, sync_fn)` |
| `auto_round/algorithms/quantization/sign_round/quantizer.py` | Captures return, calls `sync_gradients()` before `_step()` |

## Venv

Python: `/home/yiliu7/workspace/venvs/ar/bin/python`
