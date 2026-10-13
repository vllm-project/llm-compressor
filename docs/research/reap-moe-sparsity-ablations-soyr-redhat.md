---
title: "REAP expert pruning across MoE sizes and architectures"
tags:
  - REAP
  - MoE
  - pruning
  - calibration
  - GLM-4.7-Flash
  - Moonlight-16B
---

<div class="entry-byline" markdown>
[![soyr-redhat](https://github.com/soyr-redhat.png?size=64){ .entry-avatar }](https://github.com/soyr-redhat)
[![kylesayrs](https://github.com/kylesayrs.png?size=64){ .entry-avatar }](https://github.com/kylesayrs)
**[@soyr-redhat](https://github.com/soyr-redhat)** and **[@kylesayrs](https://github.com/kylesayrs)** posted on **2026-08-25**, updated **2026-10-09**
</div>

# REAP expert pruning across MoE sizes and architectures

## Summary

REAP at 25% sparsity is close to free on GLM-4.7-Flash: 98 to 100% quality
recovery on standard tasks, 20 to 22% higher offline generation throughput,
and 23% smaller weights. Three findings were less expected. Safe sparsity
tracks expert count rather than model size, the calibration dataset matters
roughly an order of magnitude more than the calibration sample count, and
"free at 25%" stops being true once the eval moves past loglikelihood and
short-form generation onto knowledge and reasoning tasks.

## Setup

- Models: [`zai-org/GLM-4.7-Flash`](https://huggingface.co/zai-org/GLM-4.7-Flash)
  (bf16, 47 layers, 64 routed experts, top-4) and
  [`moonshotai/Moonlight-16B-A3B-Instruct`](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct)
  (bf16, 27 layers, 64 routed experts, top-6)
- Recipe: `REAPPruningModifier(sparsity=0.25 / 0.50)`, prune only with no
  quantization stage, sequential pipeline, `moe_calibrate_all_experts=False`,
  saved with `save_compressed=True`
- Calibration: `HuggingFaceH4/ultrachat_200k` (UC) at 512 samples and
  `mlabonne/open-perfectblend` (PB) at 1024 samples
- Hardware: 1x H100 80GB, vLLM with TP=1, `max_model_len=4096`
- Throughput: offline `LLM.generate`, 128 prompts x 256 output tokens,
  `ignore_eos`, whole-generation wall clock, n=1

## Results: standard tasks

GLM-4.7-Flash, reporting gsm8k strict, arc_challenge acc_norm, and hellaswag
acc_norm.

| Variant | gsm8k | arc_c | hellaswag | Avg recovery | out tok/s | weights |
|---|---|---|---|---|---|---|
| Baseline (64e) | 0.8355 | 0.5751 | 0.8023 | - | 5495 | 55.9 GiB |
| REAP-25 UC (48e) | 0.8279 | 0.5751 | 0.7931 | **0.9931** | 6704 (+22.0%) | 42.9 GiB |
| REAP-25 PB (48e) | 0.8218 | 0.5538 | 0.7862 | 0.9755 | 6600 (+20.1%) | 42.9 GiB |
| REAP-50 UC (32e) | 0.5914 | 0.5282 | 0.7393 | 0.8492 | 8096 (+47.3%) | 30.0 GiB |
| REAP-50 PB (32e) | 0.7748 | 0.4437 | 0.6341 | 0.8298 | 8271 (+50.5%) | 30.0 GiB |

Moonlight-16B-A3B-Instruct, UC calibration.

| Variant | gsm8k | arc_c | hellaswag | Avg recovery | out tok/s | weights |
|---|---|---|---|---|---|---|
| Baseline (64e) | 0.7498 | 0.5828 | 0.7882 | - | 10516 | 29.8 GiB |
| REAP-25 UC (48e) | 0.6308 | 0.5555 | 0.7854 | 0.9303 | 11618 (+10.5%) | 23.1 GiB |
| REAP-50 UC (32e) | 0.1463 | 0.4386 | 0.7413 | 0.6294 | 13819 (+31.4%) | 16.4 GiB |

## Results: reasoning tasks

Same checkpoints, `minerva_math500` (math_verify) and GPQA Diamond CoT
zeroshot (flexible-extract), `max_model_len=8192`.

| GLM-4.7-Flash variant | GPQA | recovery | MATH-500 | recovery |
|---|---|---|---|---|
| Baseline (64e) | 0.3939 | - | 0.7360 | - |
| REAP-25 UC (48e) | 0.3384 | 85.9% | 0.6380 | 86.7% |
| REAP-25 PB (48e) | 0.3333 | 84.6% | 0.6920 | **94.0%** |
| REAP-50 UC (32e) | 0.1111 | 28.2% | 0.4340 | 59.0% |
| REAP-50 PB (32e) | 0.1465 | 37.2% | 0.5660 | **76.9%** |

Moonlight MATH-500 recovery was 37.2% at REAP-25 and 10.1% at REAP-50. GPQA is
omitted for Moonlight because its baseline of 0.2020 sits below the 25% random
floor, so the task carries no signal for that model.

## Results: calibration dataset versus sample count

GLM-4.7-Flash, REAP-50, gsm8k strict.

| | 512 samples | 1024 samples | count effect |
|---|---|---|---|
| ultrachat_200k | 0.5914 | 0.6285 | +3.7pp |
| open-perfectblend | 0.7612 | 0.7748 | +1.4pp |
| **dataset effect** | **+17.0pp** | **+14.6pp** | |

## Findings

"Avg recovery" above is the unweighted mean of the per-task ratios against
baseline, so a single collapsed task can be masked by two healthy ones. Read
the per-task columns before trusting the average.

### Expert count, not parameter count, predicts the safe sparsity

The two models here have identical expert counts and very different
tolerances, which rules out parameter count as the explanation.

| | GLM-4.7-Flash (64e) | Moonlight-16B (64e) | Qwen3-30B-A3B (128e) |
|---|---|---|---|
| REAP-25 | 99.1% | 84.2% | 98.9% |
| REAP-50 | 92.7% (PB) | 19.5% | 99.8% |

Qwen3-30B-A3B numbers are from `examples/reap_expert_pruning`, not run here.
Moonlight is the smallest model in the table and degrades worst, but it is
also top-6 routing against GLM's top-4, so each token already activates more
of the expert pool and there is less idle capacity to reclaim. The practical
reading is that you cannot infer a safe sparsity from model size. Pilot the
specific checkpoint, and treat a high expert count as the thing that buys
headroom.

### Calibration data choice outweighs calibration volume

The dataset arm moves gsm8k recovery by 14.6 to 17.0pp. The sample-count arm
moves it by 1.4 to 3.7pp, so roughly an order of magnitude less for twice the
calibration cost. On a fixed budget, spend it on choosing the dataset.

One caveat that does **not** apply here but will bite anyone scaling this up.
The default `data_collator="truncation"` truncates every batch to its shortest
member, so with a large `batch_size` most calibration tokens are silently
discarded. llm-compressor warns about this above `batch_size=16`. My runs left
`batch_size` at its default of 1, where a batch is its own shortest member and
nothing is lost, so the 512 versus 1024 comparison is clean. Kyle's
`batch_size=64` runs saw roughly 75k effective tokens against 680k, which is
why his report-to-prune comparison is confounded. If you raise the batch size
for calibration speed, your effective sample count is not the number you
passed in.

### The dataset choice trades task families against each other

Perfectblend is not strictly better. At REAP-50 it gains 15 to 17pp on gsm8k
while giving up 8.4pp on arc_challenge and 10.5pp on hellaswag. The averages
nearly cancel (0.8298 PB against 0.8492 UC) and hide a large reallocation
underneath. Match the calibration mixture to the deployment workload rather
than reaching for the one with the best headline number.

### Degradation is task-asymmetric, in a consistent four-tier order

Collecting every task at REAP-25 on GLM gives a stable hierarchy:

| Tier | Tasks | Recovery at REAP-25 |
|---|---|---|
| Loglikelihood | hellaswag, arc_challenge | 99 to 100% |
| Short-form generative | gsm8k | 99% |
| Knowledge-heavy | MMLU | 92 to 94% |
| Frontier reasoning | MATH-500, GPQA | 85 to 94% |

"REAP-25 is free" is true for the top two tiers and false for the bottom two.
The gap widens sharply with sparsity: at REAP-50, GPQA falls to 28 to 37%
regardless of calibration set, an order of magnitude worse than loglikelihood
tasks at the same sparsity. If your deployment is knowledge or reasoning
heavy, benchmark it directly. gsm8k will not warn you.

### Throughput gains are sublinear in sparsity and vary by architecture

Removing 25% of experts buys 22% throughput on GLM but only 10.5% on
Moonlight; at 50% the figures are 47 to 50% against 31%. Gains are sublinear
because attention, embeddings, and the shared expert path do not shrink, and
Moonlight's top-6 routing means a larger share of its runtime was never in the
prunable pool. Throughput improvement is not predictable from the sparsity
figure alone.

### Weight savings lag expert savings

Removing 25% of experts cut GLM weights from 55.9 to 42.9 GiB, which is 23%,
not 25%, because non-expert parameters are untouched. Moonlight shows the same
23% at REAP-25. At REAP-50 the figures are 46% and 45% against 50% nominal.
Budget deployment memory from measured checkpoint size, not from the sparsity
setting.

### Two knobs that do not matter, and one that does at the margin

From the replication pass:

- **`moe_calibrate_all_experts` is a no-op for REAP.** True and False select
  identical experts in all 46 layers and produce bit-identical checkpoints
  across all 5075 tensors. REAP saliency only counts routed tokens, so the
  extra calibration work buys nothing. Leave it off and save the time.
- **`propagate_error=False` is mostly free.** It changes expert selection
  (96.4% overlap, only 7 of 46 layers identical) but lands within noise on
  every task except MMLU at REAP-50, where it is +1.6pp at roughly 2.9 sigma.
  It skips the propagation pass, so it is slightly faster.

### Know the eval noise floor before reading these tables

Because the two `moe_calibrate_all_experts` checkpoints are bit-identical, the
spread between their eval rows is pure harness nondeterminism, which gives a
measured floor: roughly 0.2pp on gsm8k and hellaswag, 2pp on MATH-500, and
2.5pp on GPQA, the last being a 198-question set. Several differences in the
reasoning tables are at or below that floor and should not be over-read.

### Operational issues worth knowing before you start

- **Set `max_gen_toks` explicitly for reasoning evals.** lm-eval generates 256
  tokens by default, which truncates chain of thought before the answer marker
  and yields meaningless scores. GPQA read 0.11 for the GLM baseline until I
  raised it to 4096, after which the same checkpoint scored 0.3939. This is a
  harness artifact that looks exactly like catastrophic pruning damage.
- **Pruned GLM-4.7-Flash checkpoints may not load in vLLM.** `resave_config`
  writes the original `n_routed_experts: 64` back into `config.json`, so vLLM
  fails on a shape mismatch. `NUM_EXPERTS_CONFIG_KEYS` is matched against
  `config.to_dict()`, but for `Glm4MoeLiteConfig` the key `num_local_experts`
  is only an `attribute_map` alias, so it never appears and the patch is
  skipped. Workaround is to set `n_routed_experts` by hand from the number of
  saved expert tensors.
- **GPQA Diamond is gated.** The official `Idavidrein/gpqa` requires access
  approval; the runs here used a mirror task against `nmayorga7/gpqa_diamond`
  with identical 198-question content and schema.

## Independent replication

[@kylesayrs replicated](https://github.com/vllm-project/llm-compressor/issues/3094#issuecomment-6083181509)
the GLM-4.7-Flash baseline and REAP-50 UC rows on 2026-10-09, added MMLU, and
ran the ablations described above. Loglikelihood and gsm8k matched closely;
GPQA and MATH-500 came in about 3pp low, which is within the noise floor that
pass established. His MMLU results, which are the source of the knowledge-heavy
tier above:

| Variant | Experts | MMLU | recovery |
|---|---|---|---|
| Baseline | 64 | 0.7083 | - |
| REAP-20 PB, torchrun | 52 | 0.6840 | 96.6% |
| REAP-25 PB, report to prune | 48 | 0.6629 | 93.6% |
| REAP-25 PB, torchrun | 48 | 0.6530 | 92.2% |
| REAP-50 UC, `propagate_error=False` | 32 | 0.5711 | 80.6% |
| REAP-50 UC | 32 | 0.5549 | 78.3% |

Within MMLU, humanities degrades most and social sciences least: at REAP-50
they retain 74.6% and 82.1% respectively. Note that the REAP-50 rows use UC at
512 samples while the REAP-20/25 rows use PB at 2048, so this is not a clean
sparsity sweep. Full tables and his environment pins are in the linked
comment.

## Reproducing

The [REAP compression, eval, and throughput scripts](https://gist.github.com/soyr-redhat/287a37158f4f5dbef56e7642dfab7e4a)
used for these runs are published as a gist. Full tables, including the
cross-architecture comparison, are in
[issue #3094](https://github.com/vllm-project/llm-compressor/issues/3094).
