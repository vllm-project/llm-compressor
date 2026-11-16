---
title: "Arabic, English or mixed calibration for W4A16 GPTQ, NVFP4 and FP8 on Qwen3-8B"
tags:
  - GPTQ
  - W4A16
  - NVFP4
  - FP8
  - Qwen3
  - calibration
  - Arabic
  - multilingual
  - DGX Spark
---

<div class="entry-byline" markdown>
[![mahmoudzahran01](https://github.com/mahmoudzahran01.png?size=64){ .entry-avatar }](https://github.com/mahmoudzahran01)
**[@mahmoudzahran01](https://github.com/mahmoudzahran01)** posted on **2026-11-08**
</div>

# Arabic, English or mixed calibration for W4A16 GPTQ, NVFP4 and FP8 on Qwen3-8B

## Summary

Does the calibration language matter when you quantize Qwen3-8B for Arabic users? In this
run it did not, at least not by an amount I could measure. Switching from English to Arabic or
to a 50/50 mix moved ArabicMMLU by at most 0.37 points for W4A16 GPTQ and NVFP4, and no
contrast was significant after Holm correction. A second English draw on its own moved W4A16
by 0.94 points. The scheme mattered far more than the language: NVFP4 lost about 4.8% of BF16
ArabicMMLU accuracy, W4A16 about 1.1%, and FP8 dynamic was within noise of BF16. On a DGX
Spark, NVFP4 had the highest throughput at concurrency 16.

## Setup

- Model: `Qwen/Qwen3-8B`, revision `b968826d9c`, BF16 baseline
- Schemes, all with `targets="Linear"`, `ignore=["lm_head"]` and preset defaults otherwise:
  - `FP8_DYNAMIC` with `QuantizationModifier`, no calibration data
  - `W4A16` with `GPTQModifier`: INT4 symmetric, group size 128, block size 128, dampening
    0.01, static activation ordering
  - `NVFP4` with `QuantizationModifier`: FP4 weights and activations in groups of 16 with FP8
    scales. The calibration data only sets the per-tensor activation global scales.
- Calibration: four sets, each 512 samples of exactly 2048 tokens (1,048,576 tokens per set).
  Each conversation is rendered with Qwen3's chat template (`enable_thinking=False`, which has no
  effect on a finished conversation, and no system prompt). A set's conversations are shuffled, concatenated and cut into 2048-token samples, so
  there is no padding and every set has the same token budget.

  | Set | Source | Licence | Seed | Conversations used | Arabic letter share |
  |-----|--------|---------|------|--------------------|---------------------|
  | en | `HuggingFaceH4/ultrachat_200k`, `train_sft` | MIT | 13 | 879 | 0.00 |
  | ar | `CohereLabs/aya_dataset`, `train`, `language_code == "arb"` | Apache-2.0 | 13 | 4,066 | 0.90 |
  | mix | first 256 samples of en and of ar, interleaved | MIT / Apache-2.0 | 13 | subset of en and ar | 0.45 |
  | en_s2 | `ultrachat_200k`, disjoint from en | MIT | 2 | 864 | 0.00 |

  Arabic rows also had to be at least 60% Arabic letters, which dropped 8 of 4,995 rows. Of the
  4,066 Aya rows used, 2,215 are original human annotations and 1,851 are human re-annotations of
  existing prompt/completion pairs.
- Hardware: one NVIDIA DGX Spark (GB10, 128 GB unified memory, driver 580.159.03). The machine
  was shared: vLLM found about 51 GiB of the 121.7 GiB it reports as device memory free at
  start-up, so it ran with `gpu_memory_utilization=0.25` (about 30 GiB).
- Quantization: llm-compressor 0.15.0, compressed-tensors 0.20.1, transformers 5.21.0, torch 2.14.1
- Eval and serving: `nvcr.io/nvidia/vllm:26.10-py3`
  (`sha256:0a0de9362545e3e062750bec4ad4f722422a04a81182b5258a7deda4325e9313`) with vLLM
  0.31.0+nv26.10, compressed-tensors 0.17.0 and transformers 5.17.0, plus lm-eval 0.4.13 on top
  (local image `sha256:89b32148cb93`)
- Evaluation: lm-eval `arabicmmlu` (40 subtasks, 14,455 questions) and `mmlu` (57 subtasks,
  14,042 questions), test splits, 0-shot, log-likelihood multiple choice, no chat template,
  vLLM backend with `max_model_len=4096` and seed 13. Accuracy is the mean of per-question
  `acc` over all questions, computed from `--log_samples`.
- Statistics, fixed before any model was quantized: paired bootstrap over questions (10,000
  resamples, seed 13, 95% percentile CI) and Holm correction over the four primary ArabicMMLU
  contrasts (W4A16 and NVFP4, each ar vs en and mix vs en). The MMLU contrasts are guards and
  are not corrected.
- Noise floor: a second W4A16 model calibrated on a disjoint English draw (en_s2). The bootstrap
  only covers which test questions were sampled, not which calibration samples were drawn, so
  this replicate is the yardstick for the language effect.

## Results

### Accuracy and recovery vs BF16

Recovery is accuracy divided by BF16 accuracy, with a paired-bootstrap 95% CI. Quantization time
is wall time including model load and save.

| Configuration | ArabicMMLU | Recovery | MMLU | Recovery | Size (GB) | Quantization time |
|---------------|-----------:|----------|-----:|----------|----------:|------------------:|
| BF16 baseline | 62.47 | - | 72.95 | - | 16.4 | - |
| FP8 dynamic | 62.36 | 99.8% [99.3, 100.3] | 72.81 | 99.8% [99.4, 100.2] | 9.44 | 2 min |
| W4A16, en | 61.79 | 98.9% [98.0, 99.8] | 71.58 | 98.1% [97.5, 98.7] | 6.07 | 33 min |
| W4A16, ar | 61.66 | 98.7% [97.9, 99.5] | 70.83 | 97.1% [96.4, 97.8] | 6.07 | 33 min |
| W4A16, mix | 61.57 | 98.6% [97.8, 99.3] | 71.77 | 98.4% [97.8, 99.0] | 6.07 | 33 min |
| W4A16, en_s2 | 60.85 | 97.4% [96.6, 98.2] | 71.54 | 98.1% [97.5, 98.7] | 6.07 | 33 min |
| NVFP4, en | 59.50 | 95.2% [94.3, 96.2] | 70.35 | 96.4% [95.8, 97.1] | 6.40 | 19 min |
| NVFP4, ar | 59.62 | 95.4% [94.5, 96.4] | 70.55 | 96.7% [96.0, 97.4] | 6.40 | 19 min |
| NVFP4, mix | 59.87 | 95.8% [94.9, 96.8] | 70.97 | 97.3% [96.6, 98.0] | 6.40 | 19 min |

### Calibration ablation: paired contrasts vs English calibration

Same scheme, same recipe, same token budget; only the calibration set changes.

| Contrast | ArabicMMLU delta (pt) [95% CI] | Holm p | MMLU delta (pt) [95% CI] | MMLU p (uncorrected) |
|----------|--------------------------------|-------:|--------------------------|---------------------:|
| W4A16, ar - en | -0.13 [-0.68, +0.43] | 1.00 | -0.75 [-1.24, -0.24] | 0.004 |
| W4A16, mix - en | -0.22 [-0.76, +0.33] | 1.00 | +0.19 [-0.28, +0.66] | 0.43 |
| NVFP4, ar - en | +0.12 [-0.46, +0.71] | 1.00 | +0.20 [-0.28, +0.68] | 0.42 |
| NVFP4, mix - en | +0.37 [-0.22, +0.93] | 0.91 | +0.61 [+0.13, +1.10] | 0.016 |
| Noise floor: W4A16, en_s2 - en | -0.94 [-1.47, -0.40] | p = 0.0002, not in the Holm family | -0.04 [-0.51, +0.44] | 0.91 |

The Arabic set shares a word 8-gram with 96 of the 14,455 ArabicMMLU questions (0.66%). As a
sensitivity check I dropped every question that shares an 8-gram with any calibration set and
reran the contrasts. That check runs over the harness's logged question text rather than the raw
dataset fields, so it matched 95 ArabicMMLU questions (and 14 MMLU questions). No delta moved by
more than 0.04 points.

### Speed

`vllm bench serve`, random dataset, 1024 input and 256 output tokens, `--ignore-eos`, seed 13.
Concurrency 1 used 32 prompts and concurrency 16 used 128, after 8 discarded warm-up prompts.
I benchmarked the English-calibrated checkpoints only, since the calibration set does not change
the kernels.

| Configuration | vLLM linear kernel | c=1 output tok/s | c=1 TTFT p50 (ms) | c=1 TPOT p50 (ms) | c=16 output tok/s | c=16 TTFT p50 (ms) | c=16 TPOT p50 (ms) |
|---------------|--------------------|----------:|------------------:|------------------:|-----------:|-------------------:|-------------------:|
| BF16 | - | 13.3 | 229 | 74.5 | 176.6 | 1181 | 87.1 |
| FP8 dynamic | `CutlassFP8ScaledMMLinearKernel` | 24.1 | 136 | 41.3 | 281.7 | 691 | 54.8 |
| W4A16 | `MarlinLinearKernel` | 39.4 | 269 | 24.7 | 342.6 | 1321 | 43.5 |
| NVFP4 | `FlashInferCutlassNvFp4LinearKernel` | 34.2 | 94 | 29.0 | 389.4 | 450 | 39.8 |

## Observations

### No calibration-language effect beyond the noise (negative result)

My hypothesis, written down before the run, was that Arabic or mixed calibration recovers more
ArabicMMLU accuracy than English for W4A16 and NVFP4 without costing MMLU accuracy. It was not
supported for either scheme. The largest language effect, +0.37 points for NVFP4 mix, is
smaller than the 0.94-point gap between two English draws.

I expected GPTQ, which uses the calibration activations for every weight update, to react more
than NVFP4. On ArabicMMLU neither moved.

### Arabic-only calibration cost English accuracy with W4A16

The one clear calibration effect was on the guard task: Arabic-only calibration lowered W4A16
MMLU by 0.75 points [-1.24, -0.24], and the drop held after the overlap filter. The English
re-draw moved MMLU by only 0.04 points, but that is a single comparison. The 50/50 mix did not
show the drop. So with GPTQ, Arabic-only calibration bought no Arabic accuracy and cost some
English, while the mix kept English intact.

NVFP4 mix scored +0.61 points on MMLU [+0.13, +1.10] (uncorrected p = 0.016). The guards are not
corrected and there is one draw per set, so I would treat it as a lead rather than a finding.

### The calibration draw mattered more than its language

This surprised me most. Two English draws of the same size from the same dataset differed by
0.94 points on ArabicMMLU (bootstrap p = 0.0002), while MMLU did not move. A question-level
bootstrap misses this source of variance, so with one draw per arm a "significant" difference
between two calibration sets can be draw noise. Use at least two draws per arm.

### The scheme mattered much more than the language

Across all calibration sets, NVFP4 lost 4.2-4.8% of BF16 ArabicMMLU accuracy and W4A16 lost
1.1-2.6%. With English calibration NVFP4 lost relatively more on Arabic than on English (4.8% vs
3.6%), while W4A16 lost less on Arabic than on English (1.1% vs 1.9%). I did not test this
difference formally. FP8 dynamic recovered 99.8% on both tasks, with both CIs including 100%.

### Speed on GB10

At concurrency 16, NVFP4 had the highest throughput: 389 output tok/s, 2.2x BF16, against 1.9x
for W4A16 and 1.6x for FP8. In a single stream W4A16 decoded fastest (39.4 tok/s, 2.9x BF16).
W4A16 prefill was slower than BF16, though: TTFT p50 was 269 ms against 229 ms at concurrency 1,
and 1.3 s at concurrency 16. NVFP4 had the lowest TTFT in both settings. Checkpoints written
with compressed-tensors 0.20.1 loaded without problems in the vLLM image, which ships
compressed-tensors 0.17.0.

### Harness note: exit code 133 at vLLM teardown

10 of the 12 lm-eval processes (in-process `vllm` model type, including smoke tests) exited with
code 133 while the vLLM engine shut down, after the last task had finished. Seven of them printed
`terminate called without an active exception`. The results were not affected: every result
file was written before the shutdown, every full eval has all 14,455 or 14,042 questions, and
the accuracies recomputed from the per-question logs match. If your driver treats a non-zero
exit as a failed eval, check the output files instead.

### Limitations and next steps

- One model, and one draw per calibration set apart from the English re-draw. The noise floor is
  a single point, not a distribution.
- 0-shot log-likelihood multiple choice only. Generation quality, dialects and long context were
  not measured.
- The Arabic set is Aya Standard Arabic, not a domain-matched Arabic chat set. A visible share is
  classical poetry and article summaries, and the conversations are short (258 tokens on average,
  against 1,193 for UltraChat). Because samples are packed with full attention across
  conversation boundaries, an Arabic sample holds about 8 conversations and an English one about
  1.7.
- The Arabic set overlaps 0.66% of ArabicMMLU questions by word 8-grams. Removing them changed
  nothing, but the check is lexical only.
- ArabicMMLU is CC-BY-NC-4.0 (research use).
- Speed numbers are single runs on one shared GB10 at `gpu_memory_utilization=0.25`.
- Next: three or more draws per calibration set, a domain-matched Arabic chat set, a generative
  Arabic eval, AWQ, and an Arabic-first model such as ALLaM.

## Reproducing

The driver scripts and per-question logs are not public. The steps below are the full recipe.

**1. Calibration sets**

```python
import re
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")

def pack(convs, keys, seed, exclude=frozenset(), n=512, seq_len=2048):
    stream, used = [], set()
    for i in np.random.default_rng(seed).permutation(len(convs)):
        if keys[i] in exclude:
            continue
        text = tok.apply_chat_template(convs[i], tokenize=False, enable_thinking=False)
        stream += tok(text, add_special_tokens=False)["input_ids"]
        used.add(keys[i])
        if len(stream) >= n * seq_len:
            break
    return [stream[k * seq_len:(k + 1) * seq_len] for k in range(n)], used

uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                  revision="8049631c405ae6576f93f445c6b8166f76f5505a")
uc_convs = [[{"role": m["role"], "content": m["content"]} for m in r["messages"]] for r in uc]
uc_ids = list(uc["prompt_id"])
en, en_used = pack(uc_convs, uc_ids, seed=13)
en_s2, _ = pack(uc_convs, uc_ids, seed=2, exclude=en_used)

AR = re.compile(r"[ء-يٮ-ۓۺ-ۿ]")
LETTER = re.compile(r"[^\W\d_]")
def ar_share(t):
    n = len(LETTER.findall(t))
    return len(AR.findall(t)) / n if n else 0.0

aya = load_dataset("CohereLabs/aya_dataset", split="train",
                   revision="f9ea04583f02a8f86404ff6c58bf75fe637df8a2")
rows = [r for r in aya if r["language_code"] == "arb"
        and ar_share(r["inputs"] + " " + r["targets"]) >= 0.6]
ar_convs = [[{"role": "user", "content": r["inputs"]},
             {"role": "assistant", "content": r["targets"]}] for r in rows]
ar, _ = pack(ar_convs, list(range(len(ar_convs))), seed=13)

mix = [s for pair in zip(en[:256], ar[:256]) for s in pair]
```

**2. Quantization** (continues from step 1, which defines `tok` and the sample lists)

```python
import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from transformers import AutoModelForCausalLM

torch.manual_seed(13)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218", dtype="auto")
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
# NVFP4: QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
# FP8:   QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
#        then oneshot(model=model, recipe=recipe) with no dataset
samples = en  # or ar, mix, en_s2
ds = Dataset.from_dict({"input_ids": samples, "attention_mask": [[1] * 2048 for _ in samples]})
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=2048,
        num_calibration_samples=512, shuffle_calibration_samples=False)
model.save_pretrained("Qwen3-8B-W4A16-en", save_compressed=True)
tok.save_pretrained("Qwen3-8B-W4A16-en")
```

**3. Accuracy.** I called `lm_eval.simple_evaluate` once per task on a single loaded model. The
equivalent CLI is:

```bash
lm_eval --model vllm \
  --model_args pretrained=Qwen3-8B-W4A16-en,dtype=auto,gpu_memory_utilization=0.25,max_model_len=4096,seed=13 \
  --tasks arabicmmlu,mmlu --num_fewshot 0 --batch_size auto --seed 13 \
  --log_samples --output_path results/
```

For BF16 use `pretrained=Qwen/Qwen3-8B,revision=b968826d9c46dd6066d109eabc6255188de91218`.
Accuracy is the mean `acc` over all logged samples of a task. For each contrast, draw 10,000
resamples of question indices with `numpy.random.default_rng(13)`, use the same indices for both
models, take the 2.5th and 97.5th percentiles of the difference, and compute
p = min(1, 2 * min(P(delta <= 0), P(delta >= 0))).

**4. Speed**

```bash
vllm serve Qwen3-8B-W4A16-en --gpu-memory-utilization 0.25 --max-model-len 4096 --seed 13
vllm bench serve --backend vllm --endpoint /v1/completions --model Qwen3-8B-W4A16-en \
  --dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos --seed 13 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --num-prompts 128 --max-concurrency 16   # and --num-prompts 32 --max-concurrency 1
```
