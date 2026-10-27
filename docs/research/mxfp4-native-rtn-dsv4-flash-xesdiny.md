---
title: "Native MXFP4 + FP8-block after DeepSeek-V4-Flash fine-tuning: RTN versus offline R1"
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

# Native MXFP4 + FP8-block after DeepSeek-V4-Flash fine-tuning: RTN versus offline R1

## Summary

A separate public-checkpoint follow-up used the same four tasks and principal evaluation settings, and found a different task-level ordering: R1-Ceil exceeded RTN on three tasks and was lower on GPQA. The two experiments use different source weights and cannot be combined into one result.

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

### Public-checkpoint follow-up

The original private fine-tuned BF16 checkpoint was removed by its owner, so we carried out a **separate experiment** using the public `deepseek-ai/DeepSeek-V4-Flash-0731` native checkpoint. This new baseline was not fine-tuned. We decoded the official FP8/MXFP4 values into BF16, aligned the tensor structure with an ms-swift export, and excluded MTP. The resulting BF16 holds values decoded from the already-quantized official weights; it is not the original pre-quantization BF16 or a reconstruction of the private fine-tune.

The four variants are Official-Native, decoded BF16 without MTP, native RTN of decoded BF16 with floor expert scales, and offline R1 followed by native quantization with FP32 routed-expert rewrite and ceil expert scales. Non-expert tensors in the R1 checkpoint use the BF16-intermediate converter path. Official-Native is the supplied original configuration; the derived variants exclude MTP, so this is an additional configuration difference rather than a pure quantize/dequantize comparison. Both RTN and R1 are data-free and use no calibration dataset. A Hadamard block size of 128 was used for R1; rotation and scale policy differ between the recipes.

We recorded download metadata identifying upstream revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`; the local weight files have not been independently verified against that remote revision. The decoded BF16 tensor payload is 568,744,420,444 bytes and each native RTN/R1 tensor payload is 156,006,389,980 bytes; these are tensor payloads, excluding file headers and sidecars, not measured GPU-memory reductions.

The public runs used eight H20-3e GPUs (143,771 MiB each), TP8, SGLang 0.5.20 at commit [`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`](https://github.com/sgl-project/sglang/commit/94602c9c2b7cbdb8efd5c52802dac6a1c180089e), PyTorch 2.13.0+cu130, and driver 580.126.20. Native variants used Marlin and FP8 wo_a; decoded BF16 used the unquantized Triton MoE path with FP8 wo_a disabled. Context length was 16,384; KV cache was automatically selected as FP8 E4M3. These are different execution paths, so a score difference is not by itself a pure quantization effect.

### Historical fine-tune serving and evaluation

The historical fine-tune task results came from SGLang. The RTN and R1 deployments were recorded as remote **4×H20, TP4** configurations. Actual GPU placement, complete startup logs, and source overlays were not independently recovered. The BF16 records contain conflicting hardware/TP information and a different reported context length. An available deployment manifest identifies a SGLang v0.5.16 image tag, but it is not a verified runtime version for every formal endpoint. Conversion used the CPU-capable checkpoint Converter workflow; a complete historical conversion-host software manifest was not retained.

The final RTN artifact has not undergone dedicated vLLM validation. Reusing native loaders without checkpoint-specific patches in both frameworks remains a deployment goal, not a verified cross-framework result.

We used lm-evaluation-harness through `local-chat-completions`, local task definitions/wrappers, and a Hugging Face tokenizer with chat templating. All tasks used 0-shot evaluation, `temperature=0`, up to 2,048 generated tokens, thinking disabled, and client concurrency 32. The client recorded random seed 0 and NumPy, Torch, and few-shot seeds 1234. Client `max_length=32768` is not evidence of the server's actual context limit.

| Task | Examples per run | Reported metric | Task stop strings |
|---|---:|---|---|
| GPQA Diamond | 198 | Accuracy | `</s>` |
| IFEval | 541 | Prompt-level strict accuracy | Empty list |
| GSM8K | 1,319 | Exact match, flexible-extract filter | `Question:`, `</s>`, `<\|im_end\|>` |
| MATH500 | 500 | Exact match | Empty list |

GSM8K's flexible extractor selects the last numeric regex match rather than requiring the strict `####` answer format. Local wrappers and answer extraction are part of this setup; these scores should not be assumed interchangeable with another harness configuration. The exact public reproduction bundle is not available with this report.

## Results

### Four repeated full-task runs

BF16 was evaluated on October 20, RTN on October 22, and R1-Ceil on October 23, 2026. Values are proportions, shown as mean ± population standard deviation (`ddof=0`) across four runs on the same task sets. This is repeat variability, not a confidence interval or four independent benchmark datasets.

| Variant | GPQA Diamond | IFEval strict | GSM8K flexible | MATH500 |
|---|---:|---:|---:|---:|
| BF16-FT, imperfect serving reference | 0.3649 ± 0.0169 | 0.6955 ± 0.0030 | 0.8313 ± 0.0023 | 0.3755 ± 0.0022 |
| Native-RTN | **0.5379 ± 0.0162** | **0.8512 ± 0.0093** | **0.8999 ± 0.0035** | **0.5205 ± 0.0084** |
| Native-R1-Ceil | 0.4091 ± 0.0179 | 0.7435 ± 0.0035 | 0.8685 ± 0.0022 | 0.4490 ± 0.0125 |

R1-Ceil was below RTN by 12.88, 10.77, 3.15, and 7.15 percentage points, respectively. We did not compute recovery against BF16 because its serving scores are not a clean upper bound.

Across the historical runs, the saved document, prompt, and target hashes and request arguments matched. Averaging the per-example metric values saved in the sample files reproduced the reported task scores. The local RTN and R1 tokenizer files also matched by hash. These checks did not re-grade every raw response or establish identical server execution or loaded checkpoint identities.

### Separate public-checkpoint experiment

The public experiment uses the same four tasks and metric choices, with a 16,384-token client/server context setting, 0-shot, temperature 0, at most 2,048 generated tokens, thinking disabled, and client concurrency 32. Each entry is mean ± population standard deviation across four repeats of the same task set. Task wrappers and the modified client used to disable thinking remain local and are not yet published. Across the public runs, the saved document, prompt, and target hashes and request arguments matched; averaging the saved per-example metric values reproduced the task scores. As above, this is not an independent re-grading of all raw responses or proof of identical server execution.

| Variant | GPQA Diamond | IFEval strict | GSM8K flexible | MATH500 |
|---|---:|---:|---:|---:|
| Official-Native | 0.6944 ± 0.0238 | 0.8544 ± 0.0054 | 0.9293 ± 0.0016 | 0.8655 ± 0.0046 |
| Decoded BF16, no MTP | 0.6692 ± 0.0222 | 0.8669 ± 0.0018 | 0.9325 ± 0.0008 | 0.8555 ± 0.0038 |
| Native RTN, no MTP | 0.6806 ± 0.0115 | 0.8535 ± 0.0063 | 0.9251 ± 0.0037 | 0.8370 ± 0.0030 |
| R1-Ceil, FP32 expert rewrite, no MTP | 0.6705 ± 0.0268 | 0.8544 ± 0.0053 | 0.9331 ± 0.0020 | 0.8555 ± 0.0068 |

In this public experiment, R1-Ceil was lower than RTN by 1.01 percentage points on GPQA, and higher by 0.09, 0.80, and 1.85 points on IFEval, GSM8K, and MATH500, respectively. The GPQA repeat variation is larger than some other differences; these observations do not establish statistical significance. This ordering differs from the private fine-tune experiment and does not establish a general R1 advantage.

The original native model and derived no-MTP variants are not matched on MTP configuration. The BF16 baseline is itself derived from quantized weights. Four repeats use the same task documents, not four independent datasets. The runtime differs between BF16 and native variants; these results should not be interpreted as recovery against an unavailable original BF16 model.

### Local numerical checks

The historical forward audit compared 69 selected expert blocks using real checkpoint weights and synthetic random inputs, not captured calibration activations. Mean relative output RMSE was **0.1952 for R1-Ceil** and **0.2148 for RTN**; R1-Ceil was lower in all 69 tested blocks. The saved result does not contain the complete historical invocation, so current script defaults cannot be asserted as the recovered historical seed/token-count settings.

A separate streaming native round-trip audit covered 114 shards and 33,389 quantized modules. Decoded weights and scales were preserved through the tested dequantize/requantize cycle. Packed bytes could differ because signed-zero encodings differ. This checks artifact stability; it does not make BF16 quantization lossless or prove full-model equivalence.

## Observations

### Better local error did not identify the better task recipe

In the historical fine-tune experiment, ceil rounding traded a coarser quantization grid for greater range and less clipping. It improved the tested R1 expert-block error relative to the earlier R1 floor-scale variant, but the final R1-Ceil recipe still scored below direct RTN on every evaluated task.

The two final recipes differ in rotation, norm fusion, scale policy, and their intermediate-precision path. The experiment does not isolate a causal effect of rotation alone. Local forward error was useful for diagnosing scale selection, but did not select the better deployment recipe.

### Follow-up checks use a different checkpoint

After the original BF16 checkpoint was removed by its owner, we checked a newer private fine-tune on October 26. This was a separate CPU experiment: expert 0 in layers 0, 10, and 20; eight synthetic standard-normal input vectors per case; seed 1234; float32 computation; Hadamard block size 128.

Four paths per case—RTN-floor, R1-floor with a BF16 intermediate, R1-ceil with a BF16 intermediate, and the FP32 ceil rewrite—passed 12 native serialization/reload checks. Exact local R1 relative errors were below `1e-6`. In these three cases RTN had lower local error than both ceil paths, and the two ceil paths produced different packed tensors. These checks support the implementation distinction; they neither replicate the old 69-block result nor extend the old task table to the new checkpoint. At that point no full-model task evaluation was performed on that newer private checkpoint. The public-checkpoint evaluation above is a separate experiment.

### Public artifact checks and serving observations

The public decoded BF16 output was checked across 114 shards and 34,223 tensors: 33,389 quantized weights were independently decoded and compared, and 834 other tensors were checked as pass-through or integer-cast values. A recorded comparison found that the direct RTN checkpoint produced through the original model-free entrypoint matched a separate parallel run on all 67,612 tensor payloads; the duplicate was later deleted. The R1 FP32-rewrite output passed schema/finite-value checks over 114 shards, retained 1,564 non-expert tensors byte-for-byte from its verified base, and passed reconstruction of expert 0's three matrices in all 43 layers. These checks do not establish full-model logits equivalence or correctness of every runtime kernel.

The deployed RTN and R1 variants both returned unexpected answers to some short arithmetic prompts in a small diagnostic set; alternative phrasings could yield different results. The 14-case probes are not an accuracy benchmark and do not explain the task-level score differences. The prior RTN and R1 findings for the private fine-tune remain specific to that checkpoint and the documented serving conditions.

### Limits of interpretation and reproduction

Some saved BF16 responses repeated reasoning without a usable final answer. Along with incomplete and conflicting deployment provenance, this prevents attributing the score differences to quantization alone. We do not conclude that RTN repaired BF16 generation or improved underlying model capabilities.

The private source checkpoint has since been deleted. Its remaining metadata does not allow regenerating the same weights. Historical evaluation artifacts and numerical audit records support the observations, but exact independent reproduction is limited by the unavailable source weights, local task wrappers, unpublished experimental implementation, and missing server records. Current unit tests and newer-checkpoint checks do not fill those historical gaps.

The reusable native Converter code contribution will be handled separately. Until its implementation and dependencies are published, this entry is an experimental field report, not an install-and-run native conversion guide. Together, the experiments show different observed recipe rankings for different checkpoints and serving setups. In the private fine-tune, improved local numerical error did not translate into improved task scores. The public follow-up did not reproduce that four-task ordering; neither experiment isolates R1 as the sole changed variable.
