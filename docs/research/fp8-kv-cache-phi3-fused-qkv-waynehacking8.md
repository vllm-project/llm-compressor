---
title: "FP8 KV cache on Phi-3's fused QKV projection, calibrated and served"
tags:
  - FP8
  - KV cache
  - Phi-3
  - fused QKV
  - calibration
---

<div class="entry-byline" markdown>
[![waynehacking8](https://github.com/waynehacking8.png?size=64){ .entry-avatar }](https://github.com/waynehacking8)
**[@waynehacking8](https://github.com/waynehacking8)** posted on **2026-10-17**
</div>

# FP8 KV cache on Phi-3's fused QKV projection, calibrated and served

## Summary

Static FP8 KV cache quantization works on `microsoft/Phi-3-mini-4k-instruct`, whose attention
uses one fused `qkv_proj` instead of separate `k_proj` and `v_proj`. Every one of the 32 layers
got calibrated K and V scales, and vLLM loaded exactly those scales. Accuracy stayed close to the
BF16 KV baseline: perplexity rose 1.0% and full GSM8K dropped 1.8 points. This is a datapoint for
the fused QKV question in [#2494](https://github.com/vllm-project/llm-compressor/issues/2494),
using the transformers implementation of Phi-3.

**Evidence:** scripts, data preparation and raw logs are in
[this gist](https://gist.github.com/waynehacking8/85a4734853dacf366b75703412d5bfad/306ae907e5815d51e7c6840585d00815d7395f94).

## Setup

- Model: `microsoft/Phi-3-mini-4k-instruct` at revision `f39ac1d28e`; weights stay in BF16, only
  the KV cache is quantized
- Scheme: FP8 E4M3 KV cache, static, per-tensor, symmetric
- Calibration data: 64 rows of `wikitext-2-raw-v1` train (revision `b08601e043`), picked with
  Python's `random.Random(104).sample` from rows of at least 256 characters
- Calibration length: each row left-padded to 512 tokens
- Calibration libraries: llm-compressor `f1e44d70f3`, compressed-tensors 0.19.1a20261014,
  transformers 5.20.0, torch 2.14.1+cu130 (a local AWQ-only patch was applied; this recipe does
  not run AWQ)
- Serving: vLLM `8f111b22ee` with torch 2.14.0+cu130 and compressed-tensors 0.17.0,
  `TRITON_ATTN` backend, eager mode, BF16 model dtype, seed 104
- Hardware: 1x RTX PRO 6000 Blackwell Max-Q
- Dates: calibration and perplexity on 2026-10-16, GSM8K on 2026-10-17

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,  # the 64 rows, pre-tokenized and left-padded to 512 tokens
    recipe=[
        QuantizationModifier(
            kv_cache_scheme={
                "num_bits": 8,
                "type": "float",
                "strategy": "tensor",
                "dynamic": False,
                "symmetric": True,
            }
        )
    ],
    max_seq_length=512,
    num_calibration_samples=64,
    shuffle_calibration_samples=False,
    pipeline="basic",
    use_eager_attention=True,
)
```

```bash
vllm serve <export> --dtype bfloat16 --max-model-len 4096 --seed 104 \
  --attention-backend TRITON_ATTN --enforce-eager \
  --kv-cache-dtype fp8    # baseline: --kv-cache-dtype bfloat16
```

The exact commands, with the batch and memory flags and the audit extension, are in the gist.

## Results

| KV cache | WikiText-2 perplexity | GSM8K strict match | GSM8K recovery |
|----------|-----------------------|--------------------|----------------|
| BF16 baseline | 5.571 | 0.8082 ± 0.0108 | - |
| FP8, calibrated scales | 5.626 | 0.7900 ± 0.0112 | 97.7% |

**Perplexity** covers the first 33,280 tokens of the WikiText-2 test split, with its non-empty
lines joined by blank lines.
Each of the 64 windows is 1024 tokens long. Windows advance by 512 tokens, so 33,279 tokens are
scored. The requests go to vLLM's completions endpoint with `echo` and `logprobs=1`.

**GSM8K** is the full test set of 1319 questions, 5-shot. lm-evaluation-harness 0.4.13 ran it
through vLLM's completions endpoint, greedy and without a chat template. It used task version 3
and few-shot seed 1234. Answers were capped at 512 new tokens.

**The GSM8K gap is not tested for significance.** Both scores come from single runs, and the
standard errors are per score. Per-question results were not saved, so I could not run a paired
test on the 1.8-point gap.

## Observations

- **`--kv-cache-dtype auto` is not a BF16 baseline.** vLLM switches `auto` to `fp8` when the
  checkpoint's quantization config declares a `kv_cache_scheme`
  ([source](https://github.com/vllm-project/vllm/blob/8f111b22ee433b0295043361a5cee0b0945eef34/vllm/model_executor/layers/attention/attention.py#L281-L291)).
  My first baseline used `auto` and gave the same perplexity as the FP8 run (5.6257). The
  baseline above uses `--kv-cache-dtype bfloat16`.
- **vLLM used the calibrated scales.** A vLLM worker extension read every layer's loaded
  `k_scale` and `v_scale` after startup and compared them with the calibration output: all of
  them matched. In the FP8 run every layer stored K and V as `float8_e4m3fn`; in the BF16 run
  none did. A `kv_cache_scheme` in the checkpoint does not show this on its own.
- **Calibration reached every layer.** The KV calibration hook ran 64 times in each of the 32
  layers, so the fused projection hid no layer.
- **Most calibration positions were padding.** The rows were left-padded to 512 tokens, and only
  11,972 positions held text. The KV observers see every position, padding included, so the
  scales also reflect pad-token keys and values. I did not test whether unpadded calibration
  changes the result.
- **Not tested:** dynamic or per-head KV scales, other attention backends, CUDA graphs,
  long-context evaluation, throughput or memory, and newer library versions.
