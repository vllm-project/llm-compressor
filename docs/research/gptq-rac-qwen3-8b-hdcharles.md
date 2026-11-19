---
title: "GPTQ reasoning-aware calibration on Qwen3-8B"
tags:
  - GPTQ
  - FP8_DYNAMIC
  - MXFP4A8
  - W4A16
  - Qwen3
---

<div class="entry-byline" markdown>
[![HDCHARLES](https://github.com/HDCharles.png?size=64){ .entry-avatar }](https://github.com/HDCharles)
**[@HDCharles](https://github.com/HDCharles)** posted on **2026-11-19**
</div>

# GPTQ reasoning-aware calibration on Qwen3-8B

## Summary

At an equal 112,808 active GPTQ Hessian-token budget, on-policy masked calibration improved FP8_DYNAMIC Math-Verify
accuracy by 1.22 percentage points over BF16 on this run. MXFP4A8 dataset-response masking scored 2.47 points
below BF16, while the on-policy masked MXFP4A8 result was effectively tied. These are single-generation-run
comparisons; the paired intervals do not capture generation variance.

## Setup

- Model: `Qwen/Qwen3-8B` BF16 baseline.
- Calibration set: 500 rows from `HuggingFaceH4/MATH-500`, pinned to revision
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
- Calibration conditions: full prompt plus dataset response; prompt-masked dataset response; prompt-masked BF16
  on-policy trace.
- Calibration budget: exactly 112,808 active Hessian tokens per checkpoint. Prompt context remains in the model input
  for masked arms but does not contribute to the Hessian. The masked arms include 39,470 prompt-context tokens.
- Trace generation: temperature 0.6, top-p 0.95, top-k 20, seed 42, 8,192-token response cap; 2,266,534 generated
  response tokens across 500 prompts, with 100 traces reaching the cap.
- GPTQ schemes: `FP8_DYNAMIC`, `MXFP4XFP8` (MXFP4 weights with dynamic FP8 activations), and `W4A16`.
- Hardware: 8× NVIDIA H100 80 GB HBM3 host. MXFP4A8 serving used vLLM
  `0.34.0rc1.dev948+gd2da27eb2`, compressed-tensors `0.20.1`, and the `HummingMxFp4LinearKernel`.
- Evaluation: Inspect Evals `inspect_evals/math`, using the same 4,500 held-out sample IDs for every model. The 500
  calibration IDs are excluded. Thinking is enabled; temperature 0.6, top-p 0.95, top-k 20, seed 42, and an
  8,192-token response cap are used.
- Evaluation runs: BF16 and FP8_DYNAMIC were served in the November 11 run with vLLM 0.31.0. MXFP4A8 was served on
  November 19 with vLLM `0.34.0rc1.dev948+gd2da27eb2`. The held-out IDs and generation settings match, but the
  vLLM versions differ.
- Scoring: Math-Verify `0.9.0`, parsing the full reference solution and full model completion, then verifying them.
  Unparsed predictions and sample errors count as incorrect.

## Results

| GPTQ condition | BF16 | FP8_DYNAMIC | Δ vs BF16 (95% paired CI) | MXFP4A8 | Δ vs BF16 (95% paired CI) |
|---|---:|---:|---:|---:|---:|
| Full prompt + dataset response | 3,568/4,500 (79.29%) | 3,585/4,500 (79.67%) | +0.38 pp [−0.65, +1.40] | 3,557/4,500 (79.04%) | −0.24 pp [−1.31, +0.82] |
| Dataset response, prompt masked | — | 3,605/4,500 (80.11%) | +0.82 pp [−0.20, +1.85] | 3,457/4,500 (76.82%) | −2.47 pp [−3.51, −1.42] |
| On-policy response, prompt masked | — | 3,623/4,500 (80.51%) | +1.22 pp [+0.22, +2.23] | 3,570/4,500 (79.33%) | +0.04 pp [−1.01, +1.10] |

The BF16 answer-generation run used 18,017,379 tokens (768,147 input and 17,249,232 output). The corresponding
quantized output-token totals were:

| Condition | Input tokens | Output tokens | Total answer tokens |
|---|---:|---:|---:|
| FP8_DYNAMIC full | 768,147 | 17,167,315 | 17,935,462 |
| FP8_DYNAMIC dataset masked | 768,147 | 17,314,859 | 18,083,006 |
| FP8_DYNAMIC on-policy masked | 768,147 | 17,373,760 | 18,141,907 |
| MXFP4A8 full | 768,147 | 17,542,198 | 18,310,345 |
| MXFP4A8 dataset masked | 768,147 | 17,289,201 | 18,057,348 |
| MXFP4A8 on-policy masked | 768,147 | 17,161,880 | 17,930,027 |

Each quantized model produced one answer for each of the same 4,500 questions. The next table averages
per-question output-token ratios (`quantized tokens / BF16 tokens × 100`). Median is the midpoint of the two center
observations; P90 is the nearest-rank observation. The bottom-450 set is the 450 questions with the lowest ratios
for that condition; its mean is calculated directly from those 450 percentages, with no percentile interpolation.
The pooled column divides the quantized and BF16 token sums over that same set.

| Condition | Mean ratio | Median | P90 | Mean ratio on lowest 450 | Pooled ratio on lowest 450 |
|---|---:|---:|---:|---:|---:|
| FP8_DYNAMIC full | 102.71% | 100.00% | 132.25% | 63.74% | 62.96% |
| FP8_DYNAMIC dataset masked | 103.62% | 100.00% | 133.62% | 65.76% | 65.21% |
| FP8_DYNAMIC on-policy masked | 104.66% | 100.00% | 135.99% | 65.21% | 64.06% |
| MXFP4A8 full | 103.96% | 100.00% | 136.57% | 63.68% | 62.98% |
| MXFP4A8 dataset masked | 102.57% | 100.00% | 134.54% | 62.37% | 61.15% |
| MXFP4A8 on-policy masked | 102.45% | 100.00% | 134.73% | 63.27% | 62.46% |

W4A16 calibration completed for all three data conditions, but those checkpoints were not evaluated in this run, so
there are no W4A16 accuracy or output-token results to report.

## Observations

- The on-policy masked condition had the highest FP8_DYNAMIC point estimate, but its gain over the BF16 baseline is
  small. The paired normal 95% intervals compare per-question correctness on the shared IDs and do not account for
  variability from regenerating answers.
- MXFP4A8 was served with a newer vLLM runtime than the BF16/FP8_DYNAMIC runs. Its deltas can include runtime
  differences as well as quantization effects; repeat the baseline and FP8 runs on the same vLLM build before
  treating cross-scheme differences as causal.
- MXFP4A8 full and on-policy masked were within the baseline interval; dataset-response masking reduced accuracy
  on this run.
- The old strict answer extractor scored the BF16 run at 30.16% because it failed to extract many valid answers.
  Math-Verify scored those same saved generations at 79.29%; its score is the primary accuracy result here.
- Output-token ratios average each question's ratio with equal weight, so they need not match the ratio of total
  tokens. In the bottom-decile analysis, the lowest 450 per-question ratios were selected separately for each
  condition.
- The calibration and evaluation prompts, generation settings, scorer, and exact held-out IDs are included in
  [`examples/gptq/reasoning_aware_calibration`](../../examples/gptq/reasoning_aware_calibration/README.md), together
  with scripts to regenerate traces, quantize, evaluate, score, and compare token use.
