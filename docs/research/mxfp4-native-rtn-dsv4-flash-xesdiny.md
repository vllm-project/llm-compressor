---
title: "Native MXFP4 + FP8-block after DeepSeek-V4-Flash fine-tuning: RTN versus offline R1"
tags:
  - RTN
  - MXFP4
  - FP8
  - SpinQuant
  - DeepSeek-V4
---

<div class="entry-byline" markdown>
[![xesdiny](https://github.com/xesdiny.png?size=64){ .entry-avatar }](https://github.com/xesdiny)
**[@xesdiny](https://github.com/xesdiny)** posted on **2026-10-23**, updated **2026-10-26**
</div>

# Native MXFP4 + FP8-block after DeepSeek-V4-Flash fine-tuning: RTN versus offline R1

## Summary

Direct native-format RTN scored above an offline R1 + RTN recipe on all four evaluated tasks for an internal ms-swift BF16 fine-tune of DeepSeek-V4-Flash-0731, even though the R1 recipe had lower error in the tested local expert-block forward comparisons. We retained RTN as the deployment baseline; this is a comparison of observed deployment recipes, with incomplete serving provenance, rather than evidence that rotation is generally ineffective or that quantization improves the BF16 model.

## Setup

### Motivation and native output

Fine-tuning produced updated BF16 weights. We wanted to reduce their deployment footprint while reusing DeepSeek's native loading paths. The experiment used a local implementation of the model-free Converter interface to write native tensors; it did not export a compressed-tensors checkpoint. The scope discussion and implementation overview are in [#3372](https://github.com/vllm-project/llm-compressor/issues/3372).

The source fine-tune is private and is called **BF16-FT** here. Its exact weights are unavailable for independent replication. It has no MTP layers; none were added from the reference model.

| Component | Output representation |
|---|---|
| Routed experts, `w1/w2/w3` | MXFP4 E2M1, two values per `int8` byte; one E8M0 scale per 32 values |
| Selected attention projections, including `indexer.wq_b` | FP8 E4M3 weights with E8M0 scales in 128 × 128 blocks |
| Shared experts | FP8 E4M3 weights with E8M0 scales in 128 × 128 blocks |
| Other tensors | No additional quantization; the R1 recipe still transforms residual-stream readers/writers and fuses the associated norms |

Native quantized modules use `.weight` and `.scale`. Configuration declares `expert_dtype: fp4` and `quant_method: fp8`, with `scale_fmt: ue8m0` and `weight_block_size: [128, 128]`. The FP8 configuration label does not make the routed experts FP8. Dynamic activation quantization is handled by the serving runtime; the offline conversion quantizes weights.

### Recipes and implementation details

| Variant | Weight preparation | Routed-expert E8M0 exponent rounding |
|---|---|---|
| BF16-FT | Fine-tuned BF16 weights | Not applicable |
| Native-RTN | Direct native weight quantization | Floor |
| Native-R1-Ceil | Offline residual-stream R1, then native quantization | Ceil |

Neither primary quantization recipe uses calibration data. R1 uses deterministic block-diagonal Hadamard rotations with block size 128, without learned rotations or online hooks. The architectural mapping concerns overlap with [SpinQuantConverter #3262](https://github.com/vllm-project/llm-compressor/pull/3262); this report does not claim that its experimental implementation is an upstream released API.

The historical R1-Ceil artifact was made by rewriting the routed experts of an existing R1 native checkpoint. That script reconstructed those experts from the BF16 source, fused norms and applied R1 in FP32, and quantized the FP32 results with ceil scales. It retained the other tensors from the existing checkpoint. This matters: the current local chained R1 Converter casts rotated weights back to their source dtype before quantization. A BF16 intermediate and the FP32 rewrite can produce different packed values and scales. They must not be described as byte-identical reproduction paths.

### Serving and evaluation

The reported task results came from SGLang. The RTN and R1 deployments were recorded as remote **4×H20, TP4** configurations. Actual GPU placement, complete startup logs, and source overlays were not independently recovered. The BF16 records contain conflicting hardware/TP information and a different reported context length. An available deployment manifest identifies a SGLang v0.5.16 image tag, but it is not a verified runtime version for every formal endpoint. Conversion used the CPU-capable checkpoint Converter workflow; a complete historical conversion-host software manifest was not retained.

The final RTN artifact has not undergone dedicated vLLM validation. Reusing native loaders without checkpoint-specific patches in both frameworks remains a deployment goal, not a verified cross-framework result.

We used lm-evaluation-harness through `local-chat-completions`, local task definitions/wrappers, and a Hugging Face tokenizer with chat templating. All tasks used 0-shot evaluation, `temperature=0`, up to 2,048 generated tokens, thinking disabled, and client concurrency 32. The client recorded random seed 0 and NumPy, Torch, and few-shot seeds 1234. Client `max_length=32768` is not evidence of the server's actual context limit.

| Task | Examples per run | Reported metric | Task stop strings |
|---|---:|---|---|
| GPQA Diamond | 198 | Accuracy | `</s>` |
| IFEval | 541 | Prompt-level strict accuracy | Empty list |
| GSM8K | 1,319 | Exact match, flexible-extract filter | `Question:`, `</s>`, `<\|im_end\|>` |
| MATH500 | 500 | Exact match | Empty list |

GSM8K's flexible extractor selects the last numeric regex match rather than requiring the strict `####` answer format. Local wrappers and answer extraction are part of this setup; these scores should not be assumed interchangeable with another harness configuration. The exact public reproduction bundle is not available with this entry.

## Results

### Four repeated full-task runs

BF16 was evaluated on October 20, RTN on October 22, and R1-Ceil on October 23, 2026. Values are proportions, shown as mean ± population standard deviation (`ddof=0`) across four runs on the same task sets. This is repeat variability, not a confidence interval or four independent benchmark datasets.

| Variant | GPQA Diamond | IFEval strict | GSM8K flexible | MATH500 |
|---|---:|---:|---:|---:|
| BF16-FT, imperfect serving reference | 0.3649 ± 0.0169 | 0.6955 ± 0.0030 | 0.8313 ± 0.0023 | 0.3755 ± 0.0022 |
| Native-RTN | **0.5379 ± 0.0162** | **0.8512 ± 0.0093** | **0.8999 ± 0.0035** | **0.5205 ± 0.0084** |
| Native-R1-Ceil | 0.4091 ± 0.0179 | 0.7435 ± 0.0035 | 0.8685 ± 0.0022 | 0.4490 ± 0.0125 |

R1-Ceil was below RTN by 12.88, 10.77, 3.15, and 7.15 percentage points, respectively. We did not compute recovery against BF16 because its serving scores are not a clean upper bound.

Across the 12 runs, 30,696 scored records passed document/prompt/target hash and request-argument consistency checks. Scores recomputed from saved samples matched the result files. RTN and R1 tokenizer files matched by hash. These checks establish client-input consistency, not identical server execution or independently verified loaded checkpoint identities.

### Local numerical checks

The historical forward audit compared 69 selected expert blocks using real checkpoint weights and synthetic random inputs, not captured calibration activations. Mean relative output RMSE was **0.1952 for R1-Ceil** and **0.2148 for RTN**; R1-Ceil was lower in all 69 tested blocks. The saved result does not contain the complete historical invocation, so current script defaults cannot be asserted as the recovered historical seed/token-count settings.

A separate streaming native round-trip audit covered 114 shards and 33,389 quantized modules. Decoded weights and scales were preserved through the tested dequantize/requantize cycle. Packed bytes could differ because signed-zero encodings differ. This checks artifact stability; it does not make BF16 quantization lossless or prove full-model equivalence.

## Observations

### Better local error did not identify the better task recipe

Ceil rounding trades a coarser quantization grid for greater range and less clipping. It improved the tested R1 expert-block error relative to the earlier R1 floor-scale variant, but the final R1-Ceil recipe still scored below direct RTN on every evaluated task.

The two final recipes differ in rotation, norm fusion, scale policy, and their intermediate-precision path. The experiment does not isolate a causal effect of rotation alone. Local forward error was useful for diagnosing scale selection, but did not select the better deployment recipe.

### Follow-up checks use a different checkpoint

After the original BF16 checkpoint was removed by its owner, we checked a newer private fine-tune on October 26. This was a separate CPU experiment: expert 0 in layers 0, 10, and 20; eight synthetic standard-normal input vectors per case; seed 1234; float32 computation; Hadamard block size 128.

Four paths per case—RTN-floor, R1-floor with a BF16 intermediate, R1-ceil with a BF16 intermediate, and the FP32 ceil rewrite—passed 12 native serialization/reload checks. Exact local R1 relative errors were below `1e-6`. In these three cases RTN had lower local error than both ceil paths, and the two ceil paths produced different packed tensors. These checks support the implementation distinction; they neither replicate the old 69-block result nor extend the old task table to the new checkpoint. No new full-model task evaluation was performed.

### Limits of interpretation and reproduction

Some saved BF16 responses repeated reasoning without a usable final answer. Along with incomplete and conflicting deployment provenance, this prevents attributing the score differences to quantization alone. We do not conclude that RTN repaired BF16 generation or improved underlying model capabilities.

The private source checkpoint has since been deleted. Its remaining metadata does not allow regenerating the same weights. Historical evaluation artifacts and numerical audit records support the observations, but exact independent reproduction is limited by the unavailable source weights, local task wrappers, unpublished experimental implementation, and missing server records. Current unit tests and newer-checkpoint checks do not fill those historical gaps.

The reusable native Converter code contribution will be handled separately. Until its implementation and dependencies are published, this entry is an experimental field report, not an install-and-run native conversion guide. The main finding worth preserving is the bounded negative result: improved local numerical error did not translate into improved task scores for the tested R1 recipe.
