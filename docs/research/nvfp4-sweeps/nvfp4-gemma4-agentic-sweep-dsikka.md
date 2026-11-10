---
title: "NVFP4 sweep on Gemma-4 26B-A4B and 31B for agentic reasoning"
tags:
  - NVFP4
  - GPTQ
  - AWQ
  - Gemma-4
  - MoE
  - calibration
  - observers
  - reasoning
---

<div class="entry-byline" markdown>
[![dsikka](https://github.com/dsikka.png?size=64){ .entry-avatar }](https://github.com/dsikka)
**[@dsikka](https://github.com/dsikka)** posted on **2026-11-05**
</div>

# Gemma-4 NVFP4 Evaluation Results (agentic reasoning)

## Summary

Nine NVFP4 checkpoints of `google/gemma-4-26B-A4B-it` and six measured
checkpoints of `google/gemma-4-31B-it` were scored against same-size BF16
baselines on GPQA-Diamond, MMLU-Pro, and LiveCodeBench v6. GPTQ-based recipes
led both tables, with 26B-A4B recovery ranging from 90.7% to 99.0% and 31B
recovery from 92.4% to 99.9%. The three 31B AWQ combinations were not run.

## Setup

- Models: `google/gemma-4-26B-A4B-it` and `google/gemma-4-31B-it` (BF16
  baselines), plus the `dsikka/gemma-4-*-NVFP4*` checkpoints listed below.
- Scheme/recipe: NVFP4 with `QuantizationModifier` (RTN), expanded
  `nvfp4_expanded_mse` / `nvfp4_expanded_imatrix` observers, `GPTQModifier`,
  and GPTQ combined with `AWQModifier(duo_scaling="both")`. Quantization keeps
  `lm_head`, embeddings, vision-tower, vision, and audio modules out of scope;
  the 26B-A4B recipe also ignores routers and calibrates all experts
  (`moe_calibrate_all_experts=True`). The 31B AWQ combinations were not run.
- Calibration: `perfectblend`, `train[:512]`, 512 sequences, 2048 tokens.
- Hardware: TP=1, one GPU per model. The specific accelerator model is not
  recorded in the source issue.
- Serving: vLLM 0.32.0, max-model-len 262144, GPU memory utilization at vLLM's
  0.92 default, `--reasoning-parser gemma4`, thinking enabled with
  `--default-chat-template-kwargs '{"enable_thinking": true}'`. NVFP4 runs add
  `--kv-cache-dtype fp8`; BF16 baselines do not.
- Evaluation: inspect-ai 0.3.287 with inspect-evals from the neuralmagic
  `livecodebench-v6` fork. GPQA-Diamond uses 198 questions, 3 independent
  sampling epochs averaged per question (avg@3), and 128 connections;
  MMLU-Pro uses 12,032 samples, 1 epoch, and 128 connections; LiveCodeBench v6
  uses 1,055 samples, 1 epoch, 32 connections / 32 sandboxes, 1800 s per
  sample, and `--fail-on-error 0.1`.
- Sampling/seeds: temperature 1.0, top_p 0.95, top_k 64 (vLLM server default;
  Inspect's vLLM provider drops top_k). No seeds were set. MMLU-Pro and
  LiveCodeBench are single runs; GPQA's three epochs are sampling epochs, not
  seed replicates. Quantized runs use `--max-tokens 65536`. BF16 baselines use
  131072 for GPQA and MMLU-Pro and 65536 for LiveCodeBench.
- Recovery: quantized / same-size BF16 per task, in percent, on the metric
  listed for that task (GPQA avg@3; MMLU-Pro and LiveCodeBench pass@1).

## Results

### Gemma-4 26B-A4B

| Configuration | GPQA-Diamond (avg@3) | MMLU-Pro | LiveCodeBench v6 | GPQA recovery | MMLU recovery | LCB recovery |
|---|---|---|---|---|---|---|
| `google/gemma-4-26B-A4B-it` | 80.5 | 85.3 | 87.6 | baseline | baseline | baseline |
| `...-NVFP4` | 75.8 | 82.9 | 79.4 | 94.1% | 97.2% | 90.7% |
| `...-NVFP4-MSE` | 76.4 | 83.1 | 79.9 | 95.0% | 97.4% | 91.2% |
| `...-NVFP4-IMATRIX` | 76.1 | 83.5 | 81.6 | 94.6% | 97.9% | 93.2% |
| `...-NVFP4-GPTQ` | 78.1 | 84.0 | 83.0 | 97.1% | 98.5% | 94.8% |
| `...-NVFP4-GPTQ-MSE` | 75.8 | 83.9 | 79.5 | 94.1% | 98.4% | 90.8% |
| `...-NVFP4-GPTQ-IMATRIX` | 79.6 | 83.8 | 85.4 | 99.0% | 98.3% | 97.5% |
| `...-NVFP4-GPTQ-AWQ` | 77.6 | 84.3 | 83.4 | 96.4% | 98.8% | 95.2% |
| `...-NVFP4-GPTQ-AWQ-MSE` | 78.1 | 84.4 | 83.2 | 97.1% | 98.9% | 95.0% |
| `...-NVFP4-GPTQ-AWQ-IMATRIX` | 77.4 | 84.1 | 85.9 | 96.2% | 98.5% | 98.1% |

### Gemma-4 31B

| Configuration | GPQA-Diamond (avg@3) | MMLU-Pro | LiveCodeBench v6 | GPQA recovery | MMLU recovery | LCB recovery |
|---|---|---|---|---|---|---|
| `google/gemma-4-31B-it` | 85.7 | 86.9 | 90.0 | baseline | baseline | baseline |
| `...-NVFP4` | 85.2 | 86.3 | 86.5 | 99.4% | 99.3% | 96.1% |
| `...-NVFP4-MSE` | 84.8 | 86.2 | 88.7 | 99.0% | 99.2% | 98.5% |
| `...-NVFP4-IMATRIX` | 84.5 | 86.4 | 90.0 | 98.6% | 99.5% | 99.9% |
| `...-NVFP4-GPTQ` | 83.7 | 80.3 | 88.3 | 97.6% | 92.4% | 98.1% |
| `...-NVFP4-GPTQ-MSE` | 85.4 | 86.3 | 89.5 | 99.6% | 99.3% | 99.4% |
| `...-NVFP4-GPTQ-IMATRIX` | 82.5 | 86.6 | 89.8 | 96.3% | 99.6% | 99.7% |
| `...-NVFP4-GPTQ-AWQ` | — | — | — | excluded | excluded | excluded |
| `...-NVFP4-GPTQ-AWQ-MSE` | — | — | — | excluded | excluded | excluded |
| `...-NVFP4-GPTQ-AWQ-IMATRIX` | — | — | — | excluded | excluded | excluded |

`...` expands to `dsikka/gemma-4-26B-A4B-it` or `dsikka/gemma-4-31B-it`.
Recovery above 100% means no measurable loss on that task, not a real gain.
Excluded rows were not run; em dashes mark unmeasured scores.

## Observations

### GPTQ-based recipes led both model sizes

On 26B-A4B, the best GPQA recovery was `GPTQ-IMATRIX` (99.0%), the best MMLU
recovery was `GPTQ-AWQ-MSE` (98.9%), and the best LiveCodeBench recovery was
`GPTQ-AWQ-IMATRIX` (98.1%). On 31B, `GPTQ-MSE` had the best GPQA recovery
(99.6%), `GPTQ-IMATRIX` the best MMLU recovery (99.6%), and plain `IMATRIX` the
best LiveCodeBench recovery (99.9%, matching BF16).

### 26B-A4B lost more than 31B, mostly on code

The weakest 26B-A4B cell was base NVFP4 on LiveCodeBench (79.4 vs 87.6 BF16,
90.7% recovery). Base NVFP4 was stronger on 31B, where the weakest cell moved to
plain GPTQ on MMLU-Pro (80.3 vs 86.9, 92.4% recovery). LiveCodeBench was the
most sensitive task for 26B-A4B; MMLU-Pro was the one large 31B outlier.

### Observer and AWQ effects were not uniform

The expanded observers did not behave as a single upgrade. On 26B-A4B,
`GPTQ-MSE` dropped LiveCodeBench recovery to 90.8% while `GPTQ-IMATRIX` raised
it to 97.5%. On 31B, `MSE` lifted plain GPTQ's GPQA recovery from 97.6% to
99.6%. On the 26B-A4B model, adding AWQ inside the GPTQ recipes raised
LiveCodeBench recovery (94.8% to 95.2%, and 97.5% to 98.1% with the iMatrix
observer) but lowered GPQA recovery (97.1% to 96.4%). The AWQ variants were not
run at 31B, so no cross-size AWQ conclusion is available.

### Limitations and next steps

- No seeds were set, MMLU-Pro and LiveCodeBench are single runs, and no
  uncertainty or noise floor is reported. Recipe rankings within a few tenths
  of a percentage point should be read loosely.
- The accelerator model is not recorded in the source issue.
- The three 31B AWQ combinations are excluded because they were not run; the
  26B AWQ variants were re-quantized and rerun.
- Throughput, weight size, and calibration ablations were not measured in this
  sweep.
- The source notes that `*` would mark a 1-epoch GPQA value while the 3-epoch
  average is pending, but no row in the supplied tables carries that marker, so
  all GPQA values here are treated as avg@3.

## Source

Ported from [issue #3458](https://github.com/vllm-project/llm-compressor/issues/3458),
part of the FP4 baseline benchmarking series tracked in
[#2921](https://github.com/vllm-project/llm-compressor/issues/2921).
