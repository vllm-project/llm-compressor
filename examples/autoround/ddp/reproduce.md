# Multi-GPU DDP AutoRound Reproduce

## torchrun (recommended)

### 8B

```bash
cd /home/yiliu7/workspace/llm-compressor

bash examples/autoround/ddp/launch_torchrun.sh \
  --model /storage/yiliu7/Qwen/Qwen3-8B \
  --scheme W4A16 \
  --nsamples 32 --iters 50 \
  --disable_torch_compile
```

### 235B

```bash
cd /home/yiliu7/workspace/llm-compressor

AR_DISABLE_DATASET_SUBPROCESS=1 GPUS_PER_GROUP=2 CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/yiliu7/workspace/venvs/llmc/bin/torchrun --nproc_per_node=2 --master_port=29500 \
examples/autoround/ddp/ddp_qwen3_multi_gpu_torchrun.py \
--model /storage/yiliu7/Qwen/Qwen3-235B-A22B-Instruct-2507 \
--scheme W4A16 --nsamples 32 --iters 50 --disable_torch_compile
```

## bash wrapper (dedicated GPU isolation)

```bash
cd /home/yiliu7/workspace/llm-compressor

AR_DISABLE_DATASET_SUBPROCESS=1 CUDA_VISIBLE_DEVICES=0,1,6,7 GPUS_PER_GROUP=2 NPROC=2 MASTER_PORT=29501 \
  bash examples/autoround/ddp/launch_multi_gpu.sh \
  ddp_qwen3_multi_gpu_example.py \
  --model /storage/yiliu7/Qwen/Qwen3-8B \
  --scheme W4A16 \
  --nsamples 32 --iters 50 \
  --disable_torch_compile \
  > /tmp/multi_gpu_test.log 2>&1 &
```

## Monitor

```bash
tail -f /tmp/multi_gpu_test.log
ps aux | grep ddp_qwen3_multi | grep -v grep
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
pkill -f ddp_qwen3_multi_gpu
```

## Verified

### 8B (2026-06-18)
```
quantized 7/7 layers in the block, loss iter 0: 19.067873 -> iter 0: 19.067873
[Rank 0] Quantization completed
Hello my name is Mandy I am 20 years old...
```
All 37 decoder layers quantized, identical loss across ranks, sample generation works.

### 235B (2026-06-19)
```
quantized 388/389 layers in the block, loss iter 0: 0.211156 -> iter 0: 0.211156
...
[Rank 0] Quantization completed
```
All 94 decoder layers quantized (388 Linear per MoE block), identical loss across ranks. ~25 min for 1 iter.

## Key Files

| File | Change |
|------|--------|
| `examples/autoround/ddp/ddp_qwen3_multi_gpu_torchrun.py` | torchrun example with patches |
| `examples/autoround/ddp/ddp_qwen3_multi_gpu_example.py` | bash wrapper example |
| `examples/autoround/ddp/fast_pipeline.py` | Replaces `SequentialPipeline.__call__` — no FX trace |
| `examples/autoround/ddp/launch_torchrun.sh` | torchrun launcher |
| `examples/autoround/ddp/launch_multi_gpu.sh` | bash wrapper (GPU partitioning) |
| `src/llmcompressor/modifiers/autoround/base.py` | `_get_local_gpu_group_size()` reads `GPUS_PER_GROUP` |
| `src/llmcompressor/pipelines/sequential/helpers.py` | Removed `disable_onloading()` from `trace_subgraphs` |
| `ar-py/auto_round/utils/distributed.py` | `setup_ddp_if_needed_` returns `(block, sync_fn)`; `current_device()` for NCCL |
| `ar-py/auto_round/algorithms/quantization/sign_round/quantizer.py` | Captures return, calls `sync_gradients()` before `_step()` |

## Required env vars

| Var | Value | Why |
|-----|-------|-----|
| `GPUS_PER_GROUP` | `2` | Triggers multi-GPU block dispatch + manual all_reduce sync |
| `AR_DISABLE_DATASET_SUBPROCESS` | `1` | Avoids `fork()` with CUDA context |
| `--disable_torch_compile` | flag | torch.compile can't handle cross-device tensors |

## Known issue: FX trace bottleneck

`trace_subgraphs` runs an FX trace on the full model — for 61K-module models (235B) it never finishes. The `fast_pipeline.py` module bypasses this by creating subgraphs directly from decoder layer names. This affects ALL models using `SequentialPipeline`, not just DDP. The AWQ example (`qwen3_moe_example_ddp.py`) with 30B MoE also hangs.

## Venv

Python: `/home/yiliu7/workspace/venvs/llmc/bin/python`
