---
title: "MXFP4 ceil_x observer on Qwen3-14B: rounding shared exponents up recovers accuracy lost to minmax"
tags:
  - MXFP4
  - ceil_x
  - Qwen3
  - RTN
---

<div class="entry-byline" markdown>
[![yzmkbk](https://github.com/yzmkbk.png?size=64){ .entry-avatar }](https://github.com/yzmkbk)
**[@yzmkbk](https://github.com/yzmkbk)** posted on **2026-11-10**
</div>

# MXFP4 ceil_x observer on Qwen3-14B: rounding shared exponents up recovers accuracy lost to minmax

## Summary

Replacing the MXFP4 minmax shared exponent with a ceil-based exponent
(`ceil(log2(absmax / 7.25))`, the `ceil_x` observer) improved Qwen3-14B
weight-only MXFP4 accuracy by +1.52pp on GPQA Diamond and +3.00pp on AIME
(10-round averages), recovering roughly a quarter of the BF16-to-MXFP4 gap.

## Setup

- Model(s): `Qwen/Qwen3-14B`, baseline precision BF16
- Scheme/recipe: weight-only MXFP4A16 (E2M1 elements, 32-value blocks, E8M0
  shared exponents), `QuantizationModifier(targets="Linear",
  ignore=["lm_head"])`; weight observer `ceil_x` with the fixed divisor
  `c=7.25` (now registered as `mxfp4_ceil_x` in
  [PR #3492](https://github.com/vllm-project/llm-compressor/pull/3492)),
  compared against `memoryless_minmax`
- Calibration: none, data-free RTN weight quantization
- Hardware: 8x Ascend950PR, 114GB HBM per card, used for both quantization
  and serving
- Software/serving: vLLM; version not recorded
- Evaluation: GPQA Diamond and AIME, 10 rounds per task, accuracy in percent;
  harness settings (temperature, few-shot, seeds) not recorded

## Results

### Primary tasks

Accuracy is the mean over 10 rounds. Recovery is the fraction of the
minmax-to-BF16 gap closed by ceil_x.

| Configuration | GPQA Diamond | AIME | Recovery (GPQA / AIME) |
|---------------|-------------:|-----:|-----------------------:|
| BF16 baseline | 62.83% | 71.67% | - |
| MXFP4, `memoryless_minmax` | 55.91% | 60.33% | - |
| MXFP4, `ceil_x` (c=7.25) | **57.43%** | **63.33%** | 22% / 26% |

Per-round accuracy spread widely: GPQA rounds spanned 53.54-60.10%
(minmax) and 55.56-61.11% (ceil_x); AIME rounds, which have roughly 30
answers per round, spanned 53.33-66.67% (minmax) and 50.00-73.33%
(ceil_x). ceil_x won 6 of 10 rounds on both tasks.

## Observations

### Rounding the shared exponent up trades outlier truncation for resolution

The default minmax path derives the E8M0 exponent by rounding the block
maximum down, so blocks whose scaled maximum lands in `[6, 8)` cannot
represent their largest values (E2M1 tops out at 6.0) and truncate them.
The ceil_x observer instead computes `ceil(log2(absmax / c))` with a
fixed divisor `c = 7.25`: values above `6 * 2^k` are still truncated, but
only 1-2 outliers per 32-value block are affected, while the remaining
values quantize with a halved step size. The divisor is fixed rather than
configurable because it is specific to the E2M1 element format; `7.25`
was picked from an offline sweep over `[6.0, 12.0]`.

### bfloat16 arithmetic at the exponent boundary matters

The `ceil(log2(absmax / c))` chain is evaluated in bfloat16 on purpose.
Log2 values just below an integer (for example -6.9936) round to the exact
integer under bfloat16's 8-bit mantissa, so `ceil` returns the coarser
exponent. Evaluating the same chain in float32 preserves the fraction,
returns the finer exponent for about 2% of groups, and measurably degraded
task accuracy in our runs. Reproducing this observer with other
precisions should keep this in mind.

### Improvements were consistent but modest

ceil_x improved both tasks over minmax, but roughly three quarters of the
BF16-to-MXFP4 gap remained. The gains are close to the per-round noise
floor on AIME despite the 10-round averaging.

### Limitations and next steps

Single model, single scheme; other architectures and formats (NVFP4, FP8)
were not tested. In the merged PR the observer was converged into a single
`mxfp4_ceil_x` that runs the existing MSE grid search from the
ceil_x-adjusted range. The numbers above measure the fixed heuristic
(reproducible with `observer_kwargs={"maxshrink": 0.0}`); the per-group
search refinement has not been task-evaluated yet, and comparing it
against fixed `c=7.25` is the natural follow-up. Evaluation harness
settings and vLLM version were not recorded.

## Reproducing

Recipe and per-round results are in
[PR #3492](https://github.com/vllm-project/llm-compressor/pull/3492).
The quantization recipe:

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="MXFP4A16",
    weight_observer="mxfp4_ceil_x",  # "ceil_x" at experiment time
    ignore=["lm_head"],
)
```

At experiment time the observer applied the fixed ceil_x heuristic
directly. The merged observer adds a per-group MSE grid search on top of
the same ceil_x range; pass `observer_kwargs={"maxshrink": 0.0}` to
reproduce the fixed heuristic measured above.
