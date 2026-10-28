---
title: "Native MXFP4 + FP8-block after DeepSeek-V4-Flash-0731 fine-tuning: RTN versus SpinQuant R1 + ceil"
tags:
  - RTN
  - MXFP4
  - FP8
  - SpinQuant
  - DeepSeek-V4-flash-0731
---

<div class="entry-byline" markdown>
[![xesdiny](https://github.com/xesdiny.png?size=64){ .entry-avatar }](https://github.com/xesdiny)
**[@xesdiny](https://github.com/xesdiny)** posted on **2026-10-27**
</div>

# Native MXFP4 + FP8-block after DeepSeek-V4-Flash-0731 fine-tuning: RTN versus SpinQuant R1 + ceil

## Summary

We compared native RTN with SpinQuant R1 + ceil on a private DeepSeek-V4-Flash-0731 fine-tune and, separately, on a public checkpoint. RTN scored higher on all four tasks in the private experiment. In the public experiment, SpinQuant R1 + ceil scored higher on three tasks and lower on GPQA. These experiments do not establish a general recipe ranking.

## Setup

The experiments used a local model-free Converter implementation to produce DeepSeek native checkpoints: MXFP4 routed experts, plus selected attention projections and shared experts in FP8-block. The output is not a compressed-tensors checkpoint; see [#3372](https://github.com/vllm-project/llm-compressor/issues/3372).

```text
Private BF16 fine-tune
  ├─> native RTN ──────────────────────────> RTN checkpoint
  └─> offline R1 rotation + quantization ──> SpinQuant R1 + ceil checkpoint

Public official native
  ├─> Official-Native baseline
  └─> decode, remove MTP ──> decoded BF16
       ├─> native RTN ─────────────────────> RTN checkpoint
       └─> offline R1 rotation + quantization ──> SpinQuant R1 + ceil checkpoint
```

- **Recipes:** Both are data-free. RTN uses floor-rounded MXFP4 routed-expert scales. SpinQuant R1 + ceil combines a local offline R1 Hadamard rotation with ceil-rounded MXFP4 routed-expert scales. Its routed experts are rewritten from the BF16 source in FP32; other tensors come from an R1 native base. A BF16 intermediate can produce different packed values and scales.
- **Inputs:** The private BF16 fine-tune had no MTP and is no longer available. Public decoded BF16 comes from the already-quantized official weights, not the original pre-quantization BF16. Official-Native retains MTP; the derived public variants do not.
- **Evaluation:** Four full runs of GPQA Diamond, IFEval, GSM8K, and MATH500, using lm-evaluation-harness with local wrappers, 0-shot, temperature 0, up to 2,048 generated tokens and thinking disabled. GSM8K uses flexible extraction. The same task sets were used across repeats.
- **Serving:** SGLang was used for both experiments. Historical RTN and R1 were recorded as 4×H20, TP4, but their full serving provenance was not recovered. Historical BF16 hardware and TP records conflict, and its reported context length differed. Public runs used 8×H20-3e, TP8, SGLang 0.5.20, and a 16,384-token context; native and decoded BF16 variants used different execution paths.

## Results

Values are mean ± population standard deviation across four runs of the same task sets, not confidence intervals.

### Private fine-tune

| Variant | GPQA Diamond | IFEval strict | GSM8K flexible | MATH500 |
|---|---:|---:|---:|---:|
| BF16-FT, imperfect serving reference | 0.3649 ± 0.0169 | 0.6955 ± 0.0030 | 0.8313 ± 0.0023 | 0.3755 ± 0.0022 |
| Native-RTN | **0.5379 ± 0.0162** | **0.8512 ± 0.0093** | **0.8999 ± 0.0035** | **0.5205 ± 0.0084** |
| SpinQuant R1 + ceil | 0.4091 ± 0.0179 | 0.7435 ± 0.0035 | 0.8685 ± 0.0022 | 0.4490 ± 0.0125 |

RTN scored higher on all four tasks, with its largest lead on GPQA (12.88 percentage points), and was retained as the deployment baseline. The BF16 result is not a clean upper bound because its serving provenance is incomplete and conflicting.

### Separate public-checkpoint experiment

| Variant | GPQA Diamond | IFEval strict | GSM8K flexible | MATH500 |
|---|---:|---:|---:|---:|
| Official-Native | 0.6944 ± 0.0238 | 0.8544 ± 0.0054 | 0.9293 ± 0.0016 | 0.8655 ± 0.0046 |
| Decoded BF16, no MTP | 0.6692 ± 0.0222 | 0.8669 ± 0.0018 | 0.9325 ± 0.0008 | 0.8555 ± 0.0038 |
| Native RTN, no MTP | 0.6806 ± 0.0115 | 0.8535 ± 0.0063 | 0.9251 ± 0.0037 | 0.8370 ± 0.0030 |
| SpinQuant R1 + ceil, FP32 expert rewrite, no MTP | 0.6705 ± 0.0268 | 0.8544 ± 0.0053 | 0.9331 ± 0.0020 | 0.8555 ± 0.0068 |

SpinQuant R1 + ceil scored higher than RTN on three tasks, including a 1.85-percentage-point lead on MATH500, and lower by 1.01 percentage points on GPQA. These differences do not establish statistical significance.

## Observations

- **Native format connects fine-tuning to serving.** The data-free Converter wrote the fine-tuned BF16 weights into the native MXFP4 + FP8-block layout for the tested SGLang deployments. A native round-trip preserved decoded weights and scales; public artifacts also underwent conversion checks, including sampled expert reconstruction.
- **Scale selection matters after R1.** In local expert-block comparisons with synthetic inputs, applying ceil or nearest instead of floor to the MXFP4 scale exponent reduced error on the tested R1-transformed weights. Ceil increases representable range at the cost of a coarser grid.
- **Local error did not select the best task recipe.** On the private fine-tune, R1 + ceil had lower error in the tested expert blocks, but RTN scored higher on all four tasks. The recipes also differ in rotation and intermediate precision, so their task scores do not isolate the effect of scale rounding or R1.
- **The public follow-up is a separate comparison.** Its different task-level ordering supports evaluating each checkpoint and recipe on the target tasks; it does not establish a general ranking. Decoded BF16 was derived from quantized weights, and MTP and runtime paths differ among variants.

The private fine-tuned BF16 source is no longer available. The experimental converter and evaluation wrappers are not yet published, and historical serving records are incomplete. The final checkpoints have not undergone dedicated vLLM validation.
