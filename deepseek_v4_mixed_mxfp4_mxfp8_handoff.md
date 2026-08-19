# DeepSeek V4 Flash Mixed MXFP4/MXFP8 Handoff

## Goal

Create and validate a real mixed DeepSeek V4 Flash checkpoint for vLLM:

- attention projections in MXFP8
- attention output projections in FP8 block format
- MoE experts in MXFP4
- vLLM generation works with the default `--moe-backend auto`

## Repositories

- vLLM: `/dev/shm/.tmp_yi/workspace/vllm`
- llm-compressor: `/dev/shm/.tmp_yi/workspace/llm-compressor`

## Current Code Changes

### vLLM

Modified:

```text
/dev/shm/.tmp_yi/workspace/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py
```

Change:

- compressed-tensors MXFP4 MoE now uses Marlin for `--moe-backend auto`
- Cutlass is used only when explicitly requested with `--moe-backend cutlass`

Reason:

- Cutlass failed on this checkpoint with `run_fp4_blockwise_scaled_group_mm_sm100`
  GEMM initialization failure or illegal memory access.
- Marlin passed the reduced and full smoke tests.

Untracked file to leave alone:

```text
/dev/shm/.tmp_yi/workspace/vllm/ds.v4.log.cutlass
```

### llm-compressor

Modified:

```text
/dev/shm/.tmp_yi/workspace/llm-compressor/examples/quantizing_moe/deepseek_v4_mixed_mxfp4_mxfp8.py
```

Added:

```text
/dev/shm/.tmp_yi/workspace/llm-compressor/examples/quantizing_moe/convert_deepseek_v4_mixed_mxfp4_mxfp8_for_vllm.py
/dev/shm/.tmp_yi/workspace/llm-compressor/examples/quantizing_moe/deepseek_v4_mixed_mxfp4_mxfp8.md
```

Quantization recipe:

- `q_a_proj`, `q_b_proj`, `kv_proj`, and indexer `q_b_proj`: MXFP8
- `o_a_proj`, `o_b_proj`: FP8 block
- expert `gate_proj`, `up_proj`, `down_proj`: MXFP4

Converter behavior:

- rewrites Transformers tensor names to vLLM tensor names
- normalizes DeepSeek V4 config fields for vLLM
- converts FP8 block group format to `float-quantized`
- drops `*.weight_zero_point`
- requantizes `wo_a.weight` and `wo_b.weight` to `torch.float8_e4m3fn`
  using BF16 `[128, 128]` block scales

Untracked file to leave alone:

```text
/dev/shm/.tmp_yi/workspace/llm-compressor/docs/developer/xpu-ci-coverage.md
```

## Verified Artifacts

Reduced BF16 source:

```text
/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-8L
```

Reduced quantized source:

```text
/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-8L-Mixed-MXFP4-MXFP8
```

Reduced converted vLLM checkpoint:

```text
/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-8L-Mixed-MXFP4-MXFP8-vllm
```

Full hybrid checkpoint already verified:

```text
/dev/shm/.tmp_yi/models/Yi30/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8-vllm-o-proj-nvfp4
```

The reduced converted checkpoint passed vLLM generation with default backend
selection after the vLLM patch.

## Reduced Smoke Command

```bash
cd /dev/shm/.tmp_yi/workspace/vllm

CUDA_VISIBLE_DEVICES=0,1 \
PATH=/dev/shm/.tmp_yi/workspace/vllm/.venv/bin:$PATH \
VLLM_USE_FLASHINFER_SAMPLER=0 \
/dev/shm/.tmp_yi/workspace/vllm/.venv/bin/python \
examples/basic/offline_inference/generate.py \
--model /dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-8L-Mixed-MXFP4-MXFP8-vllm \
--tensor-parallel-size 2 \
--kv-cache-dtype fp8 \
--max-model-len 2048 \
--max-num-seqs 4 \
--max-num-batched-tokens 2048 \
--gpu-memory-utilization 0.9 \
--enforce-eager \
--max-tokens 2 \
--temperature 0
```

## Full Checkpoint Steps

1. Ensure enough storage exists for the full BF16 source and quantized output.
   The BF16 source is about 530 GiB; the quantized output is about 145 GiB.
2. Stage the full BF16 source at:

```text
/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16
```

3. Run full quantization:

```bash
cd /dev/shm/.tmp_yi/workspace/llm-compressor

CUDA_VISIBLE_DEVICES=0,1 \
DSV4_BF16_MODEL_ID=/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16 \
DSV4_SAVE_DIR=/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8 \
PYTHONPATH=/dev/shm/.tmp_yi/workspace/llm-compressor/src:/dev/shm/.tmp_yi/workspace/llm-compressor/.venv/lib/python3.12/site-packages \
/dev/shm/.tmp_yi/workspace/vllm/.venv/bin/python \
examples/quantizing_moe/deepseek_v4_mixed_mxfp4_mxfp8.py
```

4. Convert in place:

```bash
cd /dev/shm/.tmp_yi/workspace/llm-compressor

DSV4_MIXED_SOURCE_DIR=/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8 \
DSV4_MIXED_OUTPUT_DIR=/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8 \
DSV4_CONVERT_IN_PLACE=1 \
PYTHONPATH=/dev/shm/.tmp_yi/workspace/llm-compressor/src:/dev/shm/.tmp_yi/workspace/llm-compressor/.venv/lib/python3.12/site-packages \
/dev/shm/.tmp_yi/workspace/vllm/.venv/bin/python \
examples/quantizing_moe/convert_deepseek_v4_mixed_mxfp4_mxfp8_for_vllm.py
```

5. Validate with vLLM generation using the same flags as the reduced smoke test,
   but point `--model` at the full converted checkpoint.

## Expected Converted Dtypes

- `wo_a.weight` and `wo_b.weight`: `torch.float8_e4m3fn`
- `wo_a.weight_scale` and `wo_b.weight_scale`: `torch.bfloat16`
- attention MXFP8 scales: `torch.uint8`
- expert MXFP4 packed weights and scales: `torch.uint8`
- no `*.weight_zero_point` tensors

## Storage Note

The reference NVFP4 checkpoint is no longer needed for the current fix path:

```text
/dev/shm/.tmp_yi/models/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8
```

It can be deleted if storage is needed. It frees about 153 GiB, but that alone
is not enough to hold the full BF16 source plus a second full converted copy.

## Open Work

- Free or provide enough storage for the full BF16 source.
- Run full quantization.
- Run in-place conversion.
- Recheck tensor dtypes and index consistency.
- Run full vLLM smoke generation with default `--moe-backend auto`.
