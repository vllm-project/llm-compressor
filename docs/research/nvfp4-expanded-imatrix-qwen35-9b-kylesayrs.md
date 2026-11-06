---
title: "NVFP4 GPTQ on Qwen3.5-9B: expanded imatrix observer vs imatrix_mse"
tags:
  - NVFP4
  - GPTQ
  - imatrix
  - observers
  - Qwen3.5-9B
---

<div class="entry-byline" markdown>
[![kylesayrs](https://github.com/kylesayrs.png?size=64){ .entry-avatar }](https://github.com/kylesayrs)
**[@kylesayrs](https://github.com/kylesayrs)** posted on **2026-11-06**
</div>

# NVFP4 GPTQ on Qwen3.5-9B: expanded imatrix observer vs imatrix_mse

## Summary

Two NVFP4 W4A4 GPTQ checkpoints of `Qwen/Qwen3.5-9B`, differing only in the
weight observer (`nvfp4_expanded_imatrix` vs `imatrix_mse`), were scored on
GSM8K, GPQA-Diamond, MathVista testmini, and MMLU-Pro against a BF16
reference. The expanded imatrix observer scored higher on all four tasks, with
the largest gain on GPQA-Diamond (0.793 vs 0.768). Together with the Gemma-4
NVFP4 sweep in
[issue #3458](https://github.com/vllm-project/llm-compressor/issues/3458),
this motivates making `nvfp4_expanded_imatrix` the default weight observer.

## Setup

- Model: `Qwen/Qwen3.5-9B`, baseline precision bfloat16. Multimodal model with
  a vision tower and an MTP head; `lm_head` and the vision tower were left in
  bfloat16, and MTP layers were not loaded for quantization (MTP calibration
  is unsupported) and were copied unchanged into the saved checkpoints.
- Scheme/recipe: `GPTQModifier`, `targets="Linear"`, `scheme="NVFP4"`,
  `ignore=["lm_head", "re:.*visual.*"]`, `actorder: static`,
  `block_size: 128`, `dampening_frac: 0.01`; weights and input activations
  4-bit float, group size 16, E4M3 scales, `tensor_group` strategy. The two
  checkpoints differ only in `weight_observer`: `nvfp4_expanded_imatrix` vs
  `imatrix_mse`.
- Calibration: `HuggingFaceH4/ultrachat_200k`, `train_sft` split, 1024
  sequences, 2048 max tokens, chat-template formatted, shuffle seed 42.
- Hardware: quantization on a single NVIDIA B200, ~11 minutes per checkpoint.
- Software: llm-compressor at commit `bdcc78fff` (branch
  `qwen35-9b-expanded-imatrix-observer`), via the
  [Qwen3.5-9B GPTQ + imatrix example](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/qwen3_5_9b_gptq_imatrix_example.py).
- Evaluation: Inspect AI on GSM8K, GPQA-Diamond, MathVista testmini, and
  MMLU-Pro, single runs. Harness and serving versions, sampling settings,
  seeds, and evaluation GPU were not recorded.

## Results

| Task | NVFP4 + GPTQ + `nvfp4_expanded_imatrix` | NVFP4 + GPTQ + `imatrix_mse` | BF16 reference | Recovery (`nvfp4_expanded_imatrix`) |
|------|-----------------------------------------|------------------------------|----------------|-------------------------------------|
| GSM8K | 0.957 | 0.948 | 0.959 | 99.8% |
| GPQA-Diamond | 0.793 | 0.768 | 0.823 | 96.3% |
| MathVista testmini | 0.844 | 0.838 | 0.856 | 98.6% |
| MMLU-Pro | 0.812 | 0.810 | 0.829 | 98.1% |

Recovery is quantized / BF16 per task, as reported by the experimenter.
Computed from the rounded scores above, the `imatrix_mse` checkpoint recovers
98.9% (GSM8K), 93.3% (GPQA-Diamond), 97.9% (MathVista testmini), and 97.7%
(MMLU-Pro). Ratios of the rounded scores can differ from the reported recovery
by up to 0.2 points (MMLU-Pro: 0.812 / 0.829 &asymp; 97.9% vs 98.1% reported),
consistent with recovery having been computed from unrounded scores.

## Observations

### The expanded observer wins every task

Score deltas (`nvfp4_expanded_imatrix` &minus; `imatrix_mse`): GSM8K +0.009,
GPQA-Diamond +0.025, MathVista testmini +0.006, MMLU-Pro +0.002. GPQA-Diamond
is at once the largest gain and the weakest recovery for both checkpoints,
matching prior sweeps where observer choice mattered most on GPQA.

### Why range expansion helps NVFP4

`nvfp4_expanded_imatrix` runs the same E[x&sup2;]-importance-weighted MSE
range search as `imatrix_mse`, but searches scaled ranges from 1.8x down to
~0.8x of the per-group min/max range in 112 steps. The FP4 E2M1 grid has a
relatively large gap between its two largest magnitudes, 4 and 6, so mapping a
group's maximum to 4 (a 1.5x range reduction) rather than 6 can reduce error;
this is the four-over-six behavior. Searching a continuum of expansions that
includes both the 1.0x and 1.5x points was previously measured to beat
checking only those two candidates
([PR #2950](https://github.com/vllm-project/llm-compressor/pull/2950),
[FourOverSix paper](https://arxiv.org/pdf/2512.02010)).

### Relation to other sweeps

- On `Qwen/Qwen3-30B-A3B`, the observer sweep in
  [nvfp4-moe-baseline-sweep-dsikka.md](nvfp4-moe-baseline-sweep-dsikka.md)
  found `nvfp4_expanded_imatrix` vs the default observer to be a trade
  (+0.062 gpqa recovery, &minus;0.020 gsm8k). This entry instead isolates the
  expansion itself (both checkpoints use importance weighting) and finds that
  GSM8K improves as well.
- The Gemma-4 NVFP4 sweep in
  [issue #3458](https://github.com/vllm-project/llm-compressor/issues/3458)
  measured GPTQ + imatrix as the strongest variant on the 26B model (99.0%
  GPQA-Diamond and 97.5% LiveCodeBench recovery). Together with this entry's
  result, that is the motivation for setting the expanded imatrix observer as
  the default.

### Limitations and next steps

- Single runs with no seed replicates and no variance reported; the
  MathVista (+0.006) and MMLU-Pro (+0.002) deltas should be read as within
  noise. GPQA-Diamond has 198 questions, and a prior sweep measured a
  &plusmn;0.049 noise floor on seed-means there, so its +0.025 delta is the
  only one that clearly exceeds typical task noise.
- Evaluation provenance (Inspect AI and serving versions, sampling settings,
  seeds, evaluation GPU) was not recorded.
- One model, one calibration set (`ultrachat_200k`, 1024 sequences), one
  recipe; `nvfp4_expanded_mse`, other calibration sets, and other model
  families were not tested. Weight size and throughput were not measured.
- Next steps: replicate with seed repeats, add `nvfp4_expanded_mse` and
  default-observer baselines, and extend to more model families before
  finalizing a default-observer change.

## Reproducing

The quantization recipe is the
[Qwen3.5-9B GPTQ + imatrix example](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/qwen3_5_9b_gptq_imatrix_example.py)
(PR #3465) with `weight_observer="nvfp4_expanded_imatrix"` as on branch
`qwen35-9b-expanded-imatrix-observer` (commit `bdcc78fff`); the `imatrix_mse`
comparison checkpoint is the same script with
`weight_observer="imatrix_mse"`, as merged to `main`. The expanded-observer
run saved an `nvfp4-pack-quantized` checkpoint with 248 packed modules and
bfloat16 `lm_head`, vision, and MTP tensors.
