---
title: "NVFP4 vs NVFP4A16 RTN baselines on a Thai 7B instruct model, paired against BF16"
tags:
  - NVFP4
  - NVFP4A16
  - RTN
  - OpenThaiGPT
  - Qwen2.5
  - Thai
  - multilingual
---

<div class="entry-byline" markdown>
[![spped2000](https://github.com/spped2000.png?size=64){ .entry-avatar }](https://github.com/spped2000)
**[@spped2000](https://github.com/spped2000)** posted on **2026-08-11**
</div>

# NVFP4 vs NVFP4A16 RTN baselines on a Thai 7B instruct model, paired against BF16

## Summary

Round-to-nearest (RTN) NVFP4 from llm-compressor 0.12.0.1 cost
`openthaigpt/openthaigpt1.5-7b-instruct` 10.2 points on IFEval-TH (prompt-level strict), 5.6 on
MATH-500-TH and 2.4 on OpenThaiEval against BF16, all significant at p < 0.05 in paired exact
McNemar tests. Data-free NVFP4A16 (FP4 weights, BF16 activations) cut the IFEval-TH and
OpenThaiEval losses to 3.7 and 0.7 points (not significant) but not the MATH-500-TH loss (5.4
points, p = 0.0080). These are RTN baselines from one DGX Spark (GB10), to read next to the GPTQ,
calibration and observer results in the [Llama-3.1-8B NVFP4 sweep](./nvfp4-llama31-8b-sweep-dsikka.md)
and the [Qwen3-30B-A3B NVFP4 sweep](./nvfp4-moe-baseline-sweep-dsikka.md).

## Setup

- Model: `openthaigpt/openthaigpt1.5-7b-instruct` (dense `Qwen2ForCausalLM`), BF16 baseline;
  revision not recorded
- Schemes (compressed-tensors presets):
    - `NVFP4` (W4A4): FP4 weights and activations, groups of 16, FP8 scales. Calibration sets
      only the per-tensor activation global scale; per-group activation scales are dynamic.
    - `NVFP4A16`: the same FP4 weight format, BF16 activations, data-free (llm-compressor ran
      its `DataFreePipeline`)
- Recipe: `QuantizationModifier(targets="Linear", scheme=..., ignore=["lm_head", "re:.*embed.*"])`
  in `oneshot`, default observers, no GPTQ, AWQ or SmoothQuant
- Calibration (NVFP4 only): 512 documents shuffled with seed 0, 256 Thai Wikipedia
  (`wikimedia/wikipedia`, `20231101.th`) and 256 English news (`abisee/cnn_dailymail` 3.0.0),
  2,000 to 4,000 characters each, raw text, truncated at 512 tokens per the written plan
  (`PREREGISTERED.md`; the run's environment was not logged)
- Software: llm-compressor 0.12.0.1 (stated in the plan; no environment dump), compressed-tensors
  0.17.1 and transformers 5.10.1 per the
  [published NVFP4A16 config](https://huggingface.co/AGIcafet/openthaigpt1.5-7b-instruct-W4A16/blob/eee9c6556f78798ce13502bca317f808633800e0/config.json);
  torch and the quantization container not recorded
- Serving: `vllm/vllm-openai:v0.25.1`, same flags for every model, seed 0, BF16 KV cache, no
  `--quantization` flag. The server logs show `FlashInferCutlassNvFp4LinearKernel` (NVFP4) and
  `MarlinNvFp4LinearKernel` (NVFP4A16). Log-likelihood and speed runs used
  `--max-model-len 8192 --max-num-seqs 4` without prefix caching; generative runs used
  `--max-model-len 32768 --max-num-seqs 4` with prefix caching.
- Evaluation: generative tasks through evalscope and the chinda-eval branch
  [`fix/thai-lcb-and-numeric-scoring`](https://github.com/spped2000/chinda-eval/tree/fix/thai-lcb-and-numeric-scoring),
  which fixes LiveCodeBench-TH and MATH-500-TH scoring (evalscope version and chinda-eval commit
  not recorded); greedy, up to 4,096 new tokens, thinking off. Log-likelihood tasks through
  lm-eval 0.4.12, no chat template, 0-shot except MMLU (5-shot, 50 per subject).
- Hardware: 1x DGX Spark, GB10 GPU (SM121, aarch64), 121 GiB unified memory (`free -g`), NVIDIA
  driver 580.142 (nvidia-smi reports CUDA 13.0)
- Runs: one quantization per scheme and one evaluation pass per model, 2026-08-10 to 2026-08-11;
  the BF16 results are shared by both comparisons

## Results

Δ is quantized minus BF16 in points. p is the exact two-sided McNemar p-value on paired items, with
b/c the counts that only BF16 / only the quantized model got right. The plan counts p < 0.05 per
task as significant.

### Thai generative tasks

| Task (n) | BF16 | NVFP4 | Δ | p (b/c) | NVFP4A16 | Δ | p (b/c) |
|---|---|---|---|---|---|---|---|
| OpenThaiEval (1,232) | 0.6396 | 0.6161 | -2.35 | 0.021 (89/60) | 0.6323 | -0.73 | 0.40 (49/40) |
| IFEval-TH, prompt-level strict (215) | 0.6279 | 0.5256 | -10.23 | 0.00094 (32/10) | 0.5907 | -3.72 | 0.20 (19/11) |
| IFEval-TH, instruction-level strict (269 graded) | 0.7472 | 0.6580 | -8.92 | not tested | 0.6952 | -5.20 | not tested |
| MATH-500-TH (500) | 0.5260 | 0.4700 | -5.60 | 0.0084 (67/39) | 0.4720 | -5.40 | 0.0080 (62/35) |
| LiveCodeBench-TH, Pass@1 (111) | 0.1802 | 0.1622 | -1.80 | 0.75 (6/4) | 0.1802 | 0.00 | 1.0 (5/5) |
| AIME24-TH (30) | 1/30 | 1/30 | 0 | - (0/0) | 1/30 | 0 | 1.0 (1/1) |

The 19 IFEval-TH prompts with a sentence-count instruction (37 of the 306 instructions) were never
graded: evalscope could not load NLTK's `punkt_tab` data and scored them 0 for every model. They
are ties, so b/c and p are unaffected, but the IFEval-TH scores are understated; over the 196
graded prompts Δ is -11.22 (NVFP4) and -4.08 (NVFP4A16). Instruction-level IFEval-TH pools the 269
graded instructions, as lm-eval's `inst_level_strict_acc` does; evalscope's report instead
averages per prompt over all 215 (69.30, 59.61 and 64.26). It is not tested because instructions
cluster within prompts. With a Bonferroni correction across the five tested tasks of each scheme
(α = 0.01), the NVFP4 IFEval-TH and both MATH-500-TH results stay significant and the NVFP4
OpenThaiEval result does not. The fixed graders make these scores incomparable with model cards.

### Log-likelihood multiple choice (lm-eval)

Accuracy (`acc`); group rows pool their tasks' items. ThaiExam v2 is a letter-scored task over the
five exams in `scb10x/thai_exam`.

| Group (n) | BF16 | NVFP4 | Δ | p (b/c) | NVFP4A16 | Δ | p (b/c) |
|---|---|---|---|---|---|---|---|
| ThaiExam v2 (565) | 0.4796 | 0.4761 | -0.35 | 0.91 (42/40) | 0.4885 | +0.88 | 0.61 (28/33) |
| Thai MC: Belebele, XNLI, XCOPA (3,890) | 0.5465 | 0.5337 | -1.29 | 0.077 (408/358) | 0.5437 | -0.28 | 0.65 (244/233) |
| English MC: HellaSwag, ARC-Challenge, WinoGrande (12,481) | 0.6046 | 0.5899 | -1.47 | 7.2e-11 (485/302) | 0.5992 | -0.54 | 0.0057 (319/252) |
| MMLU, 5-shot, 50 per subject (2,850) | 0.7488 | 0.7253 | -2.35 | 1.7e-5 (153/86) | 0.7372 | -1.16 | 0.011 (96/63) |
| All of the above (19,786) | 0.6104 | 0.5951 | -1.53 | 3.2e-12 (1088/786) | 0.6050 | -0.54 | 0.0032 (687/581) |

### Fidelity

| Metric | BF16 | NVFP4 | NVFP4A16 |
|---|---|---|---|
| Thai Wikipedia bits/byte, first 200 docs (lower is better) | 0.3949 | 0.4125 | 0.4055 |
| WikiText-2 bits/byte (lower is better) | 0.6253 | 0.6575 | 0.6388 |
| Top-1 next-token agreement with BF16, teacher-forced Thai Wikipedia (32,424 positions) | - | 87.9% | 90.3% |

Both Thai rows use the same Wikipedia dump and split as the Thai calibration documents.

### Speed of NVFP4A16

`vllm bench serve`, random prompts, `--ignore-eos`, seed 0, one discarded warm-up and three scored
repeats; each value is the median of the three runs' medians. NVFP4 speed is not reported: the
JSON files in `results/otg7b_gate/nvfp4/perf/` hold NVFP4A16 runs and the NVFP4 JSONs were never
committed, so this entry reports speed only where a JSON and its console log agree.

| Input tokens | Output tokens | Concurrency | TTFT BF16 (ms) | TTFT NVFP4A16 (ms) | TPOT BF16 (ms) | TPOT NVFP4A16 (ms) |
|---|---|---|---|---|---|---|
| 128 | 1,024 | 1 | 84.05 | 50.07 | 80.85 | 23.92 |
| 128 | 1,024 | 4 | 251.92 | 178.77 | 65.71 | 22.85 |
| 1,024 | 128 | 1 | 208.39 | 280.51 | 81.09 | 24.00 |
| 1,024 | 128 | 4 | 825.97 | 1,096.78 | 66.24 | 23.38 |

Single-stream decode rose from 12.4 to 41.8 tokens/s (3.38x), but prefill of 1,024-token prompts
was slower than BF16. vLLM loaded 14.29 GiB of BF16 weights and 5.54 GiB for each quantized model.

## Observations

### Weight-only narrowed the IFEval-TH loss but not the math loss

On the same items, in a comparison the plan did not include, NVFP4 scored below NVFP4A16 by 6.51
points on IFEval-TH (13 items right only under NVFP4, 27 only under NVFP4A16; p = 0.038, not
significant after a Bonferroni correction across these three comparisons), 1.62 on OpenThaiEval
(46/66, p = 0.072) and 0.20 on MATH-500-TH (47/48, p = 1.0). Both presets use the same data-free
weight-quantization arguments, so the arms differ in activation quantization and GEMM kernel (the
NVFP4 weights were not published or compared). In this model the math loss is consistent with
weight rounding alone; one run per side cannot separate activations from kernel.

### Compared with the RTN row of the Llama-3.1-8B sweep

Per its linked scripts, the plain `NVFP4` row of the
[Llama-3.1-8B NVFP4 sweep](./nvfp4-llama31-8b-sweep-dsikka.md)
([issue #2922](https://github.com/vllm-project/llm-compressor/issues/2922)) uses the same modifier
and scheme as the NVFP4 arm here; model, language, calibration set and harness differ, and that
sweep averages three sampled seeds. There, math_500 had the lowest RTN recovery (0.8392) and
gained most from GPTQ (0.9153, or 0.9393 with damp05-perf-blend). Here MATH-500-TH had a recovery
of 0.8935 (NVFP4) and 0.8973 (NVFP4A16), consistent with math being the weight-sensitive task.
IFEval differs: 0.9790 instruction-level recovery there against 0.8806 (NVFP4) and 0.9303
(NVFP4A16) here, over graded instructions only (no sentence-count checks), for reasons this setup
cannot isolate.

### Thai log-likelihood scores missed the generative losses

On the NVFP4 checkpoint, ThaiExam v2 moved -0.35 points (p = 0.91) and no Thai multiple-choice
group was significant, while IFEval-TH fell 10.2 points. The pooled log-likelihood loss was
significant but small (-1.53 points).

### Calibration language was not tested

The half-Thai mix rests on the hypothesis that target-language calibration protects
target-language quality. No English-only or Thai-only control was run, so this entry neither
supports nor refutes it. RTN NVFP4 calibrates only activation global scales, and NVFP4A16 uses no
data, so any effect may show more with GPTQ or iMatrix observers.

### Limitations and next steps

- RTN with default observers only: no GPTQ (available in 0.12.0.1), AWQ or
  `nvfp4_expanded_mse` / `nvfp4_expanded_imatrix` observers, and no rerun on current main.
- One quantization per scheme, one calibration set, one greedy evaluation pass per model and no
  BF16 rerun, so there is no noise floor. IFEval-TH (215 prompts, 196 graded) cannot resolve about
  3 points; AIME24-TH (30) carries no signal.
- Not recorded: model revision, quantization environment (torch, container, environment
  variables), evalscope version and chinda-eval commit. The plan's generative McNemar script is
  missing from the repository, so the generative p-values above were recomputed from the
  per-item reviews. The per-item lm-eval samples were not committed, so the log-likelihood
  p-values were recomputed from the per-task pair counts in `paired_analysis.json`.
- No NVFP4 speed (see above), and GB10 (SM121) kernels were not cross-checked on SM100.
- The decision criteria in `PREREGISTERED.md` were committed together with the results, so their
  timing cannot be verified independently. Under them both arms were rejected for release: NVFP4
  failed three of four gates (Thai generative tasks, pooled MC -1.53 against a -1.5 floor, Thai
  top-1 agreement 87.9% against 88%) and NVFP4A16 failed on MATH-500-TH. The NVFP4A16 checkpoint
  was later published for non-math use.
- Next: GPTQ and the expanded MSE and iMatrix observers on both schemes; a calibration-language
  control (English-only vs 50/50 vs Thai-only) with a calibration-dependent method, scored on the
  same Thai tasks plus an English control; BF16 reruns or seeded sampling; an SM100 cross-check.

## Reproducing

[Scripts, logs and per-item results](https://github.com/spped2000/thaillm-nvfp4-dgx-spark/tree/dc28d4019d88d76efdfbfbb6fb10f28524dbb42b),
pinned to a commit with the data these numbers come from:

- Calibration set: [`build_calib.py`](https://github.com/spped2000/thaillm-nvfp4-dgx-spark/blob/dc28d4019d88d76efdfbfbb6fb10f28524dbb42b/scripts/build_calib.py)
  wrote the committed `calib/thai_en_calib.jsonl`.
- Quantization: [`lc_nvfp4_quantize.py`](https://github.com/spped2000/thaillm-nvfp4-dgx-spark/blob/dc28d4019d88d76efdfbfbb6fb10f28524dbb42b/scripts/lc_nvfp4_quantize.py)
  with `SRC`, `OUT`, `CALIB` and `SCHEME=NVFP4` or `SCHEME=NVFP4A16` (512 samples of 512 tokens by
  default).
- Evaluation: [`otg7b_gate.sh`](https://github.com/spped2000/thaillm-nvfp4-dgx-spark/blob/dc28d4019d88d76efdfbfbb6fb10f28524dbb42b/scripts/otg7b_gate.sh)
  for BF16 and NVFP4 (log-likelihood half rerun with `otg7b_gate_lm.sh`) and
  [the NVFP4A16 gate script](https://github.com/spped2000/thaillm-nvfp4-dgx-spark/blob/dc28d4019d88d76efdfbfbb6fb10f28524dbb42b/scripts/otg7b_gate_w4a16.sh).
  They hard-code the original machine's paths.
- Results: [BF16 and NVFP4](https://github.com/spped2000/thaillm-nvfp4-dgx-spark/tree/dc28d4019d88d76efdfbfbb6fb10f28524dbb42b/results/otg7b_gate)
  and [NVFP4A16](https://github.com/spped2000/thaillm-nvfp4-dgx-spark/tree/dc28d4019d88d76efdfbfbb6fb10f28524dbb42b/results/otg7b_gate_w4a16):
  both quantization logs (in the first directory), server logs, per-item generative reviews
  (`gen_*/*/*/reviews/eval-model/*.jsonl`), lm-eval summaries (`results_*.json`),
  `paired_analysis.json` and `PREREGISTERED.md`. The per-item lm-eval samples were not committed.

Older text in that repository makes claims this entry does not repeat; its
[`CORRECTIONS.md`](https://github.com/spped2000/thaillm-nvfp4-dgx-spark/blob/dc28d4019d88d76efdfbfbb6fb10f28524dbb42b/CORRECTIONS.md)
lists them. The repository also holds a 30B study quantized with ModelOpt, which this entry does
not cover.
