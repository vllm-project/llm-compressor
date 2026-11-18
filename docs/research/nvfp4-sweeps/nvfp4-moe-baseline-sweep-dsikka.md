---
title: "NVFP4 W4A4 sweep on Qwen3-30B-A3B: 20 variants across six reasoning tasks"
tags:
  - NVFP4
  - GPTQ
  - AWQ
  - SmoothQuant
  - MoE
  - Qwen3-30B-A3B
  - calibration
  - observers
---

<div class="entry-byline" markdown>
[![dsikka](https://github.com/dsikka.png?size=64){ .entry-avatar }](https://github.com/dsikka)
**[@dsikka](https://github.com/dsikka)** posted on **2026-08-06**, updated **2026-09-21**
</div>

# Qwen3-30B-A3B NVFP4 Evaluation Results (MoE, reasoning)

**20 NVFP4 W4A4 variants** of `Qwen/Qwen3-30B-A3B`, all scored against one BF16 baseline on six reasoning tasks under one fixed serving and harness config. The variants come from two batches that vary different things — a **calibration & algorithm** batch (12 variants: RTN / GPTQ / AWQ / SmoothQuant x perfect-blend vs UltraChat) and an **observer** batch (8 variants: `nvfp4_expanded_imatrix` vs
`nvfp4_expanded_mse` x GPTQ x AWQ) — but every table below ranks all 20 together. Observer batch rows are marked &dagger; throughout.

## Setup

**Baseline** — `Qwen/Qwen3-30B-A3B` (BF16)

**Variants** — 20 NVFP4 W4A4 models, all `nvfp4-pack-quantized`, group_size 16, `tensor_group` strategy, all 48 `mlp.gate` routers and `lm_head` left unquantized. Naming is `NVFP4[-GPTQ][-AWQ|-SmoothQuant][-ultrachat|-IMATRIX|-MSE]`:

| Token | Meaning |
|-------|---------|
| `NVFP4` (bare) | Round-to-nearest weights; the calibration pass only fits activation global scales |
| `-GPTQ` | `actorder: static` on weights |
| `-AWQ` | `AWQModifier` activation smoothing |
| `-SmoothQuant` | `SmoothQuantModifier` |
| `-ultrachat` | Calibrated on UltraChat instead of the default perfect-blend mix |
| `-IMATRIX` / `-MSE` &dagger; | Weight observer `nvfp4_expanded_imatrix` / `nvfp4_expanded_mse` |

The 12 unmarked variants all use the default observer; the 8 &dagger; variants all use
the perfect-blend calibration set. `config.json` does **not** record the AWQ
step — AWQ and non-AWQ checkpoints have byte-identical configs and safetensors
headers. Confirm from `recipe.yaml` (look for the `AWQModifier` block).

**Seeds** — 1234, 2345, 3456. aime25 additionally uses 4567, 5678, 6789, 7890, 8901.

**Sampling** — temperature 0.6, top_p 0.95, top_k 20

**Serving** — vLLM v0.28.0, TP=1 (one GPU per model), max-model-len 40960, no YaRN, `--reasoning-parser qwen3`, `--gpu-memory-utilization 0.9`

**Harness** — eval-agent `--category reasoning`, num_concurrent 32, timeout 3600

**Recovery** — seed-mean(quantized) / seed-mean(BF16), per task, on the metric listed for that task. Every score in this document is a mean over seeds.

## Recommended models

Ranked by mean recovery over the five non-gsm8k tasks (mmlu_pro_chat, ifeval, math_500, gpqa_diamond, aime25).

| Rank | Model | Mean recovery (5 tasks) | 6-task mean | Worst task | Tasks &ge; 0.99 |
|------|-------|-------------------------|-------------|------------|-----------------|
| 1 | `NVFP4-GPTQ-IMATRIX` &dagger; | 0.9887 | 0.9874 | 0.9706 (aime25) | 2 / 5 |
| 2 | `NVFP4-GPTQ-AWQ-ultrachat` | 0.9849 | 0.9885 | 0.9536 (gpqa_diamond) | 2 / 5 |
| 3 | `NVFP4-AWQ-MSE` &dagger; | 0.9840 | 0.9831 | 0.9485 (gpqa_diamond) | 2 / 5 |

- `NVFP4-GPTQ-IMATRIX` &dagger; — **Highest floor, best on the hard tasks.** Its worst task (0.9706) is the best worst-task of all 20, and it is the only variant in the document to exceed BF16 on gpqa_diamond (1.0077) — the single hardest column here. Where it gives ground is mmlu_pro (0.9709, rank 15 of 20); it is last on nothing.
- `NVFP4-GPTQ-AWQ-ultrachat` — **Best aggregate, safest default.** Highest 6-task mean of any variant (0.9885) because it is the only one that also wins gsm8k (1.0066), and it holds 2nd-best mmlu_pro (0.9986) and 2nd-best aime25 (0.9883). Its soft column is gpqa (0.9536).
- `NVFP4-AWQ-MSE` &dagger; — **aime25 / math-leaning pick.** The only variant to match BF16 on aime25 (1.0000) and above BF16 on math_500 (1.0014), with a respectable mmlu_pro (0.9806) — but it pays ~6 pts of gpqa recovery (0.9485) for it.

Caveats: 1 and 2 swap places depending on whether gsm8k is counted — `GPTQ-IMATRIX` leads by 0.0038 on the five non-gsm8k tasks, `GPTQ-AWQ-ultrachat` by 0.0011 on all six. They are also from different batches, so read the gap loosely and pick on profile: `GPTQ-IMATRIX` if nothing may regress badly, `GPTQ-AWQ-ultrachat` if gsm8k-style CoT arithmetic or mmlu_pro matter. Ranks 2 and 3 are separated by 0.0009, inside the noise on every task that drives it.

**Avoid** the bottom of the table: `NVFP4-SmoothQuant` (0.9579), `NVFP4-MSE` (0.9592, worst or second-worst on four of six tasks) and `NVFP4-GPTQ-SmoothQuant` (0.9633).

## Takeaways

- **Nothing recovers BF16.** Best 0.9885, mean over 20 is 0.9746. The gap is almost entirely gpqa_diamond (0.9485) and aime25 (0.9503) — the worst task for **19 of 20** variants. ifeval (0.9923) and math_500 (0.9993) are untouched.
- **Take GPTQ.** The only axis with a consistent, resolvable effect: gsm8k +0.0204 over 10 matched pairs, 9 of 10 positive. Costs a small but well-measured −0.0062 on mmlu_pro.
- **UltraChat buys mmlu_pro (+0.0231, 6 of 6 pairs) and costs math_500 (−0.0148).** The single most reliable effect in the sweep. Use it unless math is the target.
- **The observer is a trade, not an upgrade.** `GPTQ-IMATRIX` vs `GPTQ` is +0.0618 gpqa / −0.0197 gsm8k, both past their floors, everything else flat. Pick by target metric.
- **AWQ is a coin flip; SmoothQuant is a cost.** AWQ splits 4/4 on gsm8k, mmlu_pro, math_500 and gpqa — its only resolvable cells are two gpqa losses stacked on GPTQ (−0.0593, −0.0541), so don't stack it there. SmoothQuant is negative on 4 of 4 gsm8k pairs with no upside column.
- **Recipe choice matters less than it looks.** Best member per family: GPTQ+AWQ 0.9885 > GPTQ 0.9874 > AWQ 0.9831 > RTN 0.9770 > GPTQ+SmoothQuant 0.9743 > SmoothQuant 0.9726 — and within a family, calibration/observer moves things as much (GPTQ+AWQ spans 0.9689–0.9885). Plain RTN at 0.9770 beats 11 of the other 19.
- **Read gpqa and aime25 loosely** (floors ±0.0490, ±0.0807); ifeval carries no signal at all. **mmlu_pro is the clean discriminator** — 4.2 pt spread on a ~0.2 seed std. Recovery above 1.0 means "no measurable loss," not a gain.

## Knobs to turn:

1. GPTQ — always on. Only axis with a resolvable, consistent effect (gsm8k +0.0204, 9/10).
2. Calibration set → UltraChat vs Perfectblend
3. Observer → imatrix — biggest single resolvable cell in the document (gpqa +0.0618 over plain GPTQ)
4. AWQ — given time
5. SmoothQuant — don't.

## Metrics

One metric per task everywhere in this document — the strict / exact-match variant wherever the harness emits more than one. *Noise floor* = ~95% threshold for a difference between two seed-means, in recovery points; smaller gaps are not evidence.

| Task | Metric | Samples | Seeds | Noise floor |
|------|--------|---------|-------|-------------|
| gsm8k_platinum_cot_llama | `exact_match,strict-match` | 1,209 | 3 | ±0.0111 |
| mmlu_pro_chat | `exact_match,custom-extract` | 12,032 | 3 | ±0.0036 |
| ifeval | `inst_level_strict_acc,none` | 541 | 3 | ±0.0125 |
| math_500 | `pass@k:k=1&n=1` | 500 | 3 | ±0.0177 |
| gpqa_diamond | `gpqa_pass@k:k=1` | 198 | 3 | ±0.0490 |
| aime25 | `pass@k:k=1&n=1` | 30 | 8 | ±0.0807 |

## Recovery vs BF16

Bold = top two quantized variants on that task. Observer-sweep variants (`-IMATRIX` / `-MSE`) are marked &dagger;.

| Model | gsm8k | mmlu_pro | ifeval | math_500 | gpqa | aime25 |
|-------|-------|-------|-------|-------|-------|-------|
| NVFP4-GPTQ-AWQ-ultrachat | **1.0066** | **0.9986** | 0.9955 | 0.9883 | 0.9536 | **0.9883** |
| NVFP4-GPTQ | **1.0005** | 0.9719 | 0.9875 | 1.0138 | 0.9459 | 0.9706 |
| NVFP4-GPTQ-ultrachat | 0.9927 | 0.9772 | 0.9951 | 0.9949 | 0.9742 | 0.9589 |
| NVFP4-GPTQ-AWQ | 0.9922 | 0.9437 | 0.9924 | 0.9868 | 0.9639 | 0.9530 |
| NVFP4-GPTQ-SmoothQuant | 0.9912 | 0.9675 | **1.0049** | 1.0058 | 0.9047 | 0.9059 |
| NVFP4-GPTQ-SmoothQuant-ultrachat | 0.9881 | 0.9828 | **0.9982** | 0.9766 | 0.9588 | 0.9412 |
| NVFP4-AWQ-ultrachat | 0.9863 | 0.9982 | 0.9951 | 0.9934 | 0.9485 | 0.9177 |
| NVFP4 | 0.9852 | 0.9966 | 0.9910 | 1.0014 | 0.9227 | 0.9647 |
| NVFP4-ultrachat | 0.9794 | **0.9996** | 0.9857 | 0.9898 | 0.9175 | 0.9118 |
| NVFP4-AWQ | 0.9753 | 0.9601 | 0.9969 | **1.0262** | 0.9459 | 0.9765 |
| NVFP4-SmoothQuant-ultrachat | 0.9545 | 0.9852 | 0.9857 | **1.0182** | 0.9510 | 0.9412 |
| NVFP4-SmoothQuant | 0.9226 | 0.9630 | 0.9933 | 1.0160 | 0.9407 | 0.9118 |
| NVFP4-IMATRIX &dagger; | 0.9797 | 0.9838 | 0.9888 | 0.9664 | 0.9691 | 0.9530 |
| NVFP4-MSE &dagger; | 0.9568 | 0.9562 | 0.9834 | 1.0175 | 0.9175 | 0.9236 |
| NVFP4-GPTQ-IMATRIX &dagger; | 0.9808 | 0.9709 | 0.9870 | 1.0073 | **1.0077** | 0.9706 |
| NVFP4-GPTQ-MSE &dagger; | 0.9930 | 0.9738 | 0.9928 | 0.9796 | **0.9794** | 0.9471 |
| NVFP4-AWQ-IMATRIX &dagger; | 0.9837 | 0.9766 | 0.9946 | 0.9949 | 0.9459 | 0.9824 |
| NVFP4-AWQ-MSE &dagger; | 0.9785 | 0.9806 | 0.9897 | 1.0014 | 0.9485 | **1.0000** |
| NVFP4-GPTQ-AWQ-IMATRIX &dagger; | 0.9684 | 0.9719 | 0.9977 | 0.9941 | 0.9485 | 0.9765 |
| NVFP4-GPTQ-AWQ-MSE &dagger; | 0.9927 | 0.9802 | 0.9901 | 1.0131 | 0.9253 | 0.9118 |

| Model | Mean recovery | Worst task | Tasks &ge; 0.99 |
|-------|---------------|------------|-----------------|
| NVFP4-GPTQ-AWQ-ultrachat | 0.9885 | 0.9536 (gpqa_diamond) | 3 / 6 |
| NVFP4-GPTQ-IMATRIX &dagger; | 0.9874 | 0.9706 (aime25) | 2 / 6 |
| NVFP4-AWQ-MSE &dagger; | 0.9831 | 0.9485 (gpqa_diamond) | 2 / 6 |
| NVFP4-GPTQ-ultrachat | 0.9822 | 0.9589 (aime25) | 3 / 6 |
| NVFP4-GPTQ | 0.9817 | 0.9459 (gpqa_diamond) | 2 / 6 |
| NVFP4-AWQ | 0.9802 | 0.9459 (gpqa_diamond) | 2 / 6 |
| NVFP4-AWQ-IMATRIX &dagger; | 0.9797 | 0.9459 (gpqa_diamond) | 2 / 6 |
| NVFP4-GPTQ-MSE &dagger; | 0.9776 | 0.9471 (aime25) | 2 / 6 |
| NVFP4 | 0.9770 | 0.9227 (gpqa_diamond) | 3 / 6 |
| NVFP4-GPTQ-AWQ-IMATRIX &dagger; | 0.9762 | 0.9485 (gpqa_diamond) | 2 / 6 |
| NVFP4-GPTQ-SmoothQuant-ultrachat | 0.9743 | 0.9412 (aime25) | 1 / 6 |
| NVFP4-IMATRIX &dagger; | 0.9735 | 0.9530 (aime25) | 0 / 6 |
| NVFP4-AWQ-ultrachat | 0.9732 | 0.9177 (aime25) | 3 / 6 |
| NVFP4-SmoothQuant-ultrachat | 0.9726 | 0.9412 (aime25) | 1 / 6 |
| NVFP4-GPTQ-AWQ | 0.9720 | 0.9437 (mmlu_pro_chat) | 2 / 6 |
| NVFP4-GPTQ-AWQ-MSE &dagger; | 0.9689 | 0.9118 (aime25) | 3 / 6 |
| NVFP4-ultrachat | 0.9640 | 0.9118 (aime25) | 1 / 6 |
| NVFP4-GPTQ-SmoothQuant | 0.9633 | 0.9047 (gpqa_diamond) | 3 / 6 |
| NVFP4-MSE &dagger; | 0.9592 | 0.9175 (gpqa_diamond) | 1 / 6 |
| NVFP4-SmoothQuant | 0.9579 | 0.9118 (aime25) | 2 / 6 |

## Per-task detail

### gsm8k_platinum_cot_llama

`exact_match,strict-match` · noise floor ±0.0111

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Recovery |
|------|------|------|------|------|------|
| Qwen/Qwen3-30B-A3B (BF16) | 0.9518 | - | - | - | - |
| NVFP4-GPTQ-AWQ-ultrachat | 0.9581 | 0.9628 | 0.9529 | 0.9586 | **1.0066** |
| NVFP4-GPTQ | 0.9523 | 0.9545 | 0.9537 | 0.9487 | 1.0005 |
| NVFP4-GPTQ-MSE &dagger; | 0.9451 | 0.9495 | 0.9421 | 0.9438 | 0.9930 |
| NVFP4-GPTQ-ultrachat | 0.9449 | 0.9479 | 0.9487 | 0.9380 | 0.9927 |
| NVFP4-GPTQ-AWQ-MSE &dagger; | 0.9449 | 0.9495 | 0.9462 | 0.9388 | 0.9927 |
| NVFP4-GPTQ-AWQ | 0.9443 | 0.9471 | 0.9438 | 0.9421 | 0.9922 |
| NVFP4-GPTQ-SmoothQuant | 0.9435 | 0.9462 | 0.9429 | 0.9413 | 0.9912 |
| NVFP4-GPTQ-SmoothQuant-ultrachat | 0.9405 | 0.9537 | 0.9289 | 0.9388 | 0.9881 |
| NVFP4-AWQ-ultrachat | 0.9388 | 0.9396 | 0.9313 | 0.9454 | 0.9863 |
| NVFP4 | 0.9377 | 0.9421 | 0.9404 | 0.9305 | 0.9852 |
| NVFP4-AWQ-IMATRIX &dagger; | 0.9363 | 0.9355 | 0.9272 | 0.9462 | 0.9837 |
| NVFP4-GPTQ-IMATRIX &dagger; | 0.9336 | 0.9313 | 0.9313 | 0.9380 | 0.9808 |
| NVFP4-IMATRIX &dagger; | 0.9325 | 0.9297 | 0.9297 | 0.9380 | 0.9797 |
| NVFP4-ultrachat | 0.9322 | 0.9280 | 0.9289 | 0.9396 | 0.9794 |
| NVFP4-AWQ-MSE &dagger; | 0.9313 | 0.9272 | 0.9330 | 0.9338 | 0.9785 |
| NVFP4-AWQ | 0.9283 | 0.9280 | 0.9280 | 0.9289 | 0.9753 |
| NVFP4-GPTQ-AWQ-IMATRIX &dagger; | 0.9217 | 0.9214 | 0.9198 | 0.9239 | 0.9684 |
| NVFP4-MSE &dagger; | 0.9107 | 0.9165 | 0.9049 | 0.9107 | 0.9568 |
| NVFP4-SmoothQuant-ultrachat | 0.9085 | 0.9148 | 0.9016 | 0.9090 | 0.9545 |
| NVFP4-SmoothQuant | 0.8781 | 0.8668 | 0.8801 | 0.8875 | 0.9226 |

### mmlu_pro_chat

`exact_match,custom-extract` · noise floor ±0.0036

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Recovery |
|------|------|------|------|------|------|
| Qwen/Qwen3-30B-A3B (BF16) | 0.7541 | - | - | - | - |
| NVFP4-ultrachat | 0.7538 | 0.7546 | 0.7520 | 0.7549 | **0.9996** |
| NVFP4-GPTQ-AWQ-ultrachat | 0.7531 | 0.7553 | 0.7532 | 0.7507 | 0.9986 |
| NVFP4-AWQ-ultrachat | 0.7527 | 0.7527 | 0.7514 | 0.7541 | 0.9982 |
| NVFP4 | 0.7516 | 0.7531 | 0.7525 | 0.7491 | 0.9966 |
| NVFP4-SmoothQuant-ultrachat | 0.7430 | 0.7440 | 0.7418 | 0.7431 | 0.9852 |
| NVFP4-IMATRIX &dagger; | 0.7419 | 0.7435 | 0.7399 | 0.7421 | 0.9838 |
| NVFP4-GPTQ-SmoothQuant-ultrachat | 0.7411 | 0.7415 | 0.7406 | 0.7412 | 0.9828 |
| NVFP4-AWQ-MSE &dagger; | 0.7395 | 0.7391 | 0.7389 | 0.7405 | 0.9806 |
| NVFP4-GPTQ-AWQ-MSE &dagger; | 0.7392 | 0.7375 | 0.7416 | 0.7384 | 0.9802 |
| NVFP4-GPTQ-ultrachat | 0.7369 | 0.7360 | 0.7373 | 0.7374 | 0.9772 |
| NVFP4-AWQ-IMATRIX &dagger; | 0.7364 | 0.7399 | 0.7334 | 0.7360 | 0.9766 |
| NVFP4-GPTQ-MSE &dagger; | 0.7343 | 0.7343 | 0.7320 | 0.7367 | 0.9738 |
| NVFP4-GPTQ-AWQ-IMATRIX &dagger; | 0.7329 | 0.7335 | 0.7306 | 0.7346 | 0.9719 |
| NVFP4-GPTQ | 0.7329 | 0.7300 | 0.7321 | 0.7366 | 0.9719 |
| NVFP4-GPTQ-IMATRIX &dagger; | 0.7322 | 0.7350 | 0.7306 | 0.7310 | 0.9709 |
| NVFP4-GPTQ-SmoothQuant | 0.7296 | 0.7296 | 0.7311 | 0.7281 | 0.9675 |
| NVFP4-SmoothQuant | 0.7262 | 0.7284 | 0.7252 | 0.7251 | 0.9630 |
| NVFP4-AWQ | 0.7240 | 0.7250 | 0.7232 | 0.7239 | 0.9601 |
| NVFP4-MSE &dagger; | 0.7211 | 0.7237 | 0.7195 | 0.7201 | 0.9562 |
| NVFP4-GPTQ-AWQ | 0.7116 | 0.7118 | 0.7117 | 0.7114 | 0.9437 |

### ifeval

`inst_level_strict_acc,none` · noise floor ±0.0125

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Recovery |
|------|------|------|------|------|------|
| Qwen/Qwen3-30B-A3B (BF16) | 0.8933 | - | - | - | - |
| NVFP4-GPTQ-SmoothQuant | 0.8977 | 0.8993 | 0.9077 | 0.8861 | **1.0049** |
| NVFP4-GPTQ-SmoothQuant-ultrachat | 0.8917 | 0.8933 | 0.8897 | 0.8921 | 0.9982 |
| NVFP4-GPTQ-AWQ-IMATRIX &dagger; | 0.8913 | 0.8933 | 0.8885 | 0.8921 | 0.9977 |
| NVFP4-AWQ | 0.8905 | 0.8897 | 0.8993 | 0.8825 | 0.9969 |
| NVFP4-GPTQ-AWQ-ultrachat | 0.8893 | 0.8945 | 0.8825 | 0.8909 | 0.9955 |
| NVFP4-GPTQ-ultrachat | 0.8889 | 0.8861 | 0.8897 | 0.8909 | 0.9951 |
| NVFP4-AWQ-ultrachat | 0.8889 | 0.8873 | 0.8909 | 0.8885 | 0.9951 |
| NVFP4-AWQ-IMATRIX &dagger; | 0.8885 | 0.8849 | 0.8909 | 0.8897 | 0.9946 |
| NVFP4-SmoothQuant | 0.8873 | 0.8825 | 0.8873 | 0.8921 | 0.9933 |
| NVFP4-GPTQ-MSE &dagger; | 0.8869 | 0.8849 | 0.8801 | 0.8957 | 0.9928 |
| NVFP4-GPTQ-AWQ | 0.8865 | 0.8933 | 0.8813 | 0.8849 | 0.9924 |
| NVFP4 | 0.8853 | 0.8957 | 0.8777 | 0.8825 | 0.9910 |
| NVFP4-GPTQ-AWQ-MSE &dagger; | 0.8845 | 0.8921 | 0.8849 | 0.8765 | 0.9901 |
| NVFP4-AWQ-MSE &dagger; | 0.8841 | 0.8801 | 0.8861 | 0.8861 | 0.9897 |
| NVFP4-IMATRIX &dagger; | 0.8833 | 0.8861 | 0.8813 | 0.8825 | 0.9888 |
| NVFP4-GPTQ | 0.8821 | 0.8801 | 0.8861 | 0.8801 | 0.9875 |
| NVFP4-GPTQ-IMATRIX &dagger; | 0.8817 | 0.8825 | 0.8837 | 0.8789 | 0.9870 |
| NVFP4-ultrachat | 0.8805 | 0.8825 | 0.8717 | 0.8873 | 0.9857 |
| NVFP4-SmoothQuant-ultrachat | 0.8805 | 0.8825 | 0.8741 | 0.8849 | 0.9857 |
| NVFP4-MSE &dagger; | 0.8785 | 0.8885 | 0.8741 | 0.8729 | 0.9834 |

### math_500

`pass@k:k=1&n=1` · noise floor ±0.0177

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Recovery |
|------|------|------|------|------|------|
| Qwen/Qwen3-30B-A3B (BF16) | 0.9147 | - | - | - | - |
| NVFP4-AWQ | 0.9387 | 0.9440 | 0.9320 | 0.9400 | **1.0262** |
| NVFP4-SmoothQuant-ultrachat | 0.9313 | 0.9360 | 0.9360 | 0.9220 | 1.0182 |
| NVFP4-MSE &dagger; | 0.9307 | 0.9300 | 0.9400 | 0.9220 | 1.0175 |
| NVFP4-SmoothQuant | 0.9293 | 0.9220 | 0.9260 | 0.9400 | 1.0160 |
| NVFP4-GPTQ | 0.9273 | 0.9240 | 0.9280 | 0.9300 | 1.0138 |
| NVFP4-GPTQ-AWQ-MSE &dagger; | 0.9267 | 0.9360 | 0.9240 | 0.9200 | 1.0131 |
| NVFP4-GPTQ-IMATRIX &dagger; | 0.9213 | 0.8980 | 0.9360 | 0.9300 | 1.0073 |
| NVFP4-GPTQ-SmoothQuant | 0.9200 | 0.9080 | 0.9220 | 0.9300 | 1.0058 |
| NVFP4 | 0.9160 | 0.9060 | 0.9160 | 0.9260 | 1.0014 |
| NVFP4-AWQ-MSE &dagger; | 0.9160 | 0.9200 | 0.9020 | 0.9260 | 1.0014 |
| NVFP4-GPTQ-ultrachat | 0.9100 | 0.9020 | 0.9060 | 0.9220 | 0.9949 |
| NVFP4-AWQ-IMATRIX &dagger; | 0.9100 | 0.9220 | 0.9020 | 0.9060 | 0.9949 |
| NVFP4-GPTQ-AWQ-IMATRIX &dagger; | 0.9093 | 0.9080 | 0.9180 | 0.9020 | 0.9941 |
| NVFP4-AWQ-ultrachat | 0.9087 | 0.9100 | 0.9040 | 0.9120 | 0.9934 |
| NVFP4-ultrachat | 0.9053 | 0.9180 | 0.8820 | 0.9160 | 0.9898 |
| NVFP4-GPTQ-AWQ-ultrachat | 0.9040 | 0.9080 | 0.9020 | 0.9020 | 0.9883 |
| NVFP4-GPTQ-AWQ | 0.9027 | 0.9080 | 0.9140 | 0.8860 | 0.9868 |
| NVFP4-GPTQ-MSE &dagger; | 0.8960 | 0.9000 | 0.8780 | 0.9100 | 0.9796 |
| NVFP4-GPTQ-SmoothQuant-ultrachat | 0.8933 | 0.8960 | 0.8880 | 0.8960 | 0.9766 |
| NVFP4-IMATRIX &dagger; | 0.8840 | 0.9000 | 0.8680 | 0.8840 | 0.9664 |

### gpqa_diamond

`gpqa_pass@k:k=1` · noise floor ±0.0490

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Recovery |
|------|------|------|------|------|------|
| Qwen/Qwen3-30B-A3B (BF16) | 0.6532 | - | - | - | - |
| NVFP4-GPTQ-IMATRIX &dagger; | 0.6582 | 0.6515 | 0.6414 | 0.6818 | **1.0077** |
| NVFP4-GPTQ-MSE &dagger; | 0.6397 | 0.6263 | 0.6263 | 0.6667 | 0.9794 |
| NVFP4-GPTQ-ultrachat | 0.6364 | 0.6515 | 0.6263 | 0.6313 | 0.9742 |
| NVFP4-IMATRIX &dagger; | 0.6330 | 0.6364 | 0.6414 | 0.6212 | 0.9691 |
| NVFP4-GPTQ-AWQ | 0.6296 | 0.6212 | 0.6414 | 0.6263 | 0.9639 |
| NVFP4-GPTQ-SmoothQuant-ultrachat | 0.6263 | 0.6465 | 0.6212 | 0.6111 | 0.9588 |
| NVFP4-GPTQ-AWQ-ultrachat | 0.6229 | 0.6263 | 0.6212 | 0.6212 | 0.9536 |
| NVFP4-SmoothQuant-ultrachat | 0.6212 | 0.6212 | 0.6111 | 0.6313 | 0.9510 |
| NVFP4-AWQ-ultrachat | 0.6196 | 0.6162 | 0.6263 | 0.6162 | 0.9485 |
| NVFP4-AWQ-MSE &dagger; | 0.6195 | 0.6313 | 0.6111 | 0.6162 | 0.9485 |
| NVFP4-GPTQ-AWQ-IMATRIX &dagger; | 0.6195 | 0.6515 | 0.6061 | 0.6010 | 0.9485 |
| NVFP4-GPTQ | 0.6179 | 0.6465 | 0.6061 | 0.6010 | 0.9459 |
| NVFP4-AWQ | 0.6179 | 0.5859 | 0.6111 | 0.6566 | 0.9459 |
| NVFP4-AWQ-IMATRIX &dagger; | 0.6178 | 0.6061 | 0.6364 | 0.6111 | 0.9459 |
| NVFP4-SmoothQuant | 0.6145 | 0.5909 | 0.6212 | 0.6313 | 0.9407 |
| NVFP4-GPTQ-AWQ-MSE &dagger; | 0.6044 | 0.6212 | 0.5758 | 0.6162 | 0.9253 |
| NVFP4 | 0.6027 | 0.6162 | 0.5859 | 0.6061 | 0.9227 |
| NVFP4-ultrachat | 0.5993 | 0.5808 | 0.6263 | 0.5909 | 0.9175 |
| NVFP4-MSE &dagger; | 0.5993 | 0.6010 | 0.5960 | 0.6010 | 0.9175 |
| NVFP4-GPTQ-SmoothQuant | 0.5909 | 0.5909 | 0.5657 | 0.6162 | 0.9047 |

### aime25

`pass@k:k=1&n=1` · noise floor ±0.0807

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Seed 4567 | Seed 5678 | Seed 6789 | Seed 7890 | Seed 8901 | Recovery |
|------|------|------|------|------|------|------|------|------|------|------|
| Qwen/Qwen3-30B-A3B (BF16) | 0.7083 | - | - | - | - | - | - | - | - | - |
| NVFP4-AWQ-MSE &dagger; | 0.7083 | 0.7333 | 0.6667 | 0.7000 | 0.7333 | 0.7333 | 0.7000 | 0.6667 | 0.7333 | **1.0000** |
| NVFP4-GPTQ-AWQ-ultrachat | 0.7000 | 0.7333 | 0.7333 | 0.6667 | 0.7000 | 0.6000 | 0.7333 | 0.7667 | 0.6667 | 0.9883 |
| NVFP4-AWQ-IMATRIX &dagger; | 0.6958 | 0.5333 | 0.6667 | 0.7000 | 0.7333 | 0.7000 | 0.7333 | 0.7333 | 0.7667 | 0.9824 |
| NVFP4-GPTQ-AWQ-IMATRIX &dagger; | 0.6917 | 0.7000 | 0.7333 | 0.7333 | 0.8000 | 0.7000 | 0.6000 | 0.6333 | 0.6333 | 0.9765 |
| NVFP4-AWQ | 0.6916 | 0.6333 | 0.7333 | 0.7000 | 0.7333 | 0.7000 | 0.6333 | 0.7333 | 0.6667 | 0.9765 |
| NVFP4-GPTQ | 0.6875 | 0.7333 | 0.6667 | 0.7333 | 0.6667 | 0.7333 | 0.5667 | 0.7000 | 0.7000 | 0.9706 |
| NVFP4-GPTQ-IMATRIX &dagger; | 0.6875 | 0.6667 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 0.6333 | 0.7333 | 0.6667 | 0.9706 |
| NVFP4 | 0.6833 | 0.6333 | 0.7333 | 0.7333 | 0.6333 | 0.7000 | 0.6333 | 0.6667 | 0.7333 | 0.9647 |
| NVFP4-GPTQ-ultrachat | 0.6792 | 0.7333 | 0.7333 | 0.6667 | 0.6333 | 0.6667 | 0.7000 | 0.7333 | 0.5667 | 0.9589 |
| NVFP4-GPTQ-AWQ | 0.6750 | 0.7333 | 0.7000 | 0.5667 | 0.7000 | 0.6667 | 0.7333 | 0.6667 | 0.6333 | 0.9530 |
| NVFP4-IMATRIX &dagger; | 0.6750 | 0.7333 | 0.7000 | 0.7000 | 0.6667 | 0.6667 | 0.7000 | 0.6667 | 0.5667 | 0.9530 |
| NVFP4-GPTQ-MSE &dagger; | 0.6708 | 0.6333 | 0.6000 | 0.6667 | 0.7000 | 0.7000 | 0.7000 | 0.6667 | 0.7000 | 0.9471 |
| NVFP4-GPTQ-SmoothQuant-ultrachat | 0.6667 | 0.6333 | 0.6667 | 0.6667 | 0.7333 | 0.6667 | 0.7333 | 0.5667 | 0.6667 | 0.9412 |
| NVFP4-SmoothQuant-ultrachat | 0.6667 | 0.7333 | 0.7000 | 0.7333 | 0.6667 | 0.6333 | 0.6667 | 0.5333 | 0.6667 | 0.9412 |
| NVFP4-MSE &dagger; | 0.6542 | 0.6000 | 0.6667 | 0.6667 | 0.7000 | 0.6667 | 0.6333 | 0.6000 | 0.7000 | 0.9236 |
| NVFP4-AWQ-ultrachat | 0.6500 | 0.7000 | 0.6667 | 0.5667 | 0.7333 | 0.6000 | 0.7333 | 0.5667 | 0.6333 | 0.9177 |
| NVFP4-SmoothQuant | 0.6459 | 0.6667 | 0.6667 | 0.7667 | 0.5667 | 0.5667 | 0.5667 | 0.6333 | 0.7333 | 0.9118 |
| NVFP4-GPTQ-AWQ-MSE &dagger; | 0.6458 | 0.6667 | 0.6000 | 0.7667 | 0.6667 | 0.5667 | 0.6333 | 0.6667 | 0.6000 | 0.9118 |
| NVFP4-ultrachat | 0.6458 | 0.6667 | 0.7000 | 0.6333 | 0.6667 | 0.7000 | 0.5333 | 0.6333 | 0.6333 | 0.9118 |
| NVFP4-GPTQ-SmoothQuant | 0.6417 | 0.6667 | 0.6667 | 0.6000 | 0.6333 | 0.6333 | 0.7000 | 0.6333 | 0.6000 | 0.9059 |

## Effect of adding GPTQ (`actorder: static`)

All 10 pairs across both batches. **recovery(with GPTQ) &minus; recovery(without)**, holding the rest of the recipe, the calibration set and the observer fixed; bold = exceeds that task's noise floor. Observer-batch pairs are marked &dagger;.

| Held fixed | gsm8k | mmlu_pro | ifeval | math_500 | gpqa | aime25 | Mean |
|---|---|---|---|---|---|---|---|
| AWQ / perfect-blend | **+0.0168** | **-0.0164** | -0.0045 | **-0.0394** | +0.0180 | -0.0235 | -0.0082 |
| AWQ / ultrachat | **+0.0203** | +0.0004 | +0.0004 | -0.0051 | +0.0051 | +0.0706 | +0.0153 |
| SmoothQuant / perfect-blend | **+0.0686** | **+0.0045** | +0.0116 | -0.0102 | -0.0360 | -0.0059 | +0.0054 |
| SmoothQuant / ultrachat | **+0.0336** | -0.0025 | **+0.0125** | **-0.0415** | +0.0078 | +0.0000 | +0.0017 |
| none / perfect-blend | **+0.0154** | **-0.0248** | -0.0036 | +0.0124 | +0.0232 | +0.0059 | +0.0048 |
| none / ultrachat | **+0.0133** | **-0.0225** | +0.0094 | +0.0051 | **+0.0567** | +0.0471 | +0.0182 |
| AWQ / imatrix &dagger; | **-0.0154** | **-0.0047** | +0.0031 | -0.0007 | +0.0026 | -0.0059 | -0.0035 |
| AWQ / mse &dagger; | **+0.0142** | -0.0004 | +0.0004 | +0.0117 | -0.0232 | **-0.0882** | -0.0143 |
| none / imatrix &dagger; | +0.0012 | **-0.0128** | -0.0018 | **+0.0408** | +0.0387 | +0.0176 | +0.0139 |
| none / mse &dagger; | **+0.0362** | **+0.0175** | +0.0094 | **-0.0379** | **+0.0619** | +0.0235 | +0.0184 |
| **Mean** (10 pairs) | +0.0204 | -0.0062 | +0.0037 | -0.0065 | +0.0155 | +0.0041 | +0.0052 |

## Effect of adding AWQ

All 8 pairs across both batches. **recovery(with AWQ) &minus; recovery(without)**; bold = exceeds that task's noise floor. Observer-batch pairs are marked &dagger;.

| Held fixed | gsm8k | mmlu_pro | ifeval | math_500 | gpqa | aime25 | Mean |
|---|---|---|---|---|---|---|---|
| GPTQ / perfect-blend | -0.0084 | **-0.0282** | +0.0049 | **-0.0270** | +0.0180 | -0.0176 | -0.0097 |
| GPTQ / ultrachat | **+0.0139** | **+0.0214** | +0.0004 | -0.0066 | -0.0206 | +0.0294 | +0.0063 |
| none / perfect-blend | -0.0098 | **-0.0365** | +0.0058 | **+0.0248** | +0.0232 | +0.0118 | +0.0032 |
| none / ultrachat | +0.0069 | -0.0015 | +0.0094 | +0.0036 | +0.0310 | +0.0059 | +0.0092 |
| GPTQ / imatrix &dagger; | **-0.0125** | +0.0010 | +0.0107 | -0.0131 | **-0.0593** | +0.0059 | -0.0112 |
| GPTQ / mse &dagger; | -0.0003 | **+0.0064** | -0.0027 | **+0.0335** | **-0.0541** | -0.0353 | -0.0087 |
| none / imatrix &dagger; | +0.0041 | **-0.0072** | +0.0058 | **+0.0284** | -0.0232 | +0.0294 | +0.0062 |
| none / mse &dagger; | **+0.0217** | **+0.0244** | +0.0063 | -0.0160 | +0.0309 | +0.0765 | +0.0240 |
| **Mean** (8 pairs) | +0.0020 | -0.0025 | +0.0051 | +0.0035 | -0.0068 | +0.0132 | +0.0024 |

## Effect of adding SmoothQuant

All 4 pairs; the observer batch has no SmoothQuant checkpoint. **recovery(with SmoothQuant) &minus; recovery(without)**; bold = exceeds that task's noise floor.

| Held fixed | gsm8k | mmlu_pro | ifeval | math_500 | gpqa | aime25 | Mean |
|---|---|---|---|---|---|---|---|
| GPTQ / perfect-blend | -0.0093 | **-0.0044** | **+0.0175** | -0.0080 | -0.0412 | -0.0647 | -0.0184 |
| GPTQ / ultrachat | -0.0046 | **+0.0056** | +0.0031 | **-0.0182** | -0.0155 | -0.0176 | -0.0079 |
| none / perfect-blend | **-0.0625** | **-0.0336** | +0.0022 | +0.0146 | +0.0180 | -0.0529 | -0.0190 |
| none / ultrachat | **-0.0249** | **-0.0144** | +0.0000 | **+0.0284** | +0.0335 | +0.0294 | +0.0087 |
| **Mean** (4 pairs) | -0.0253 | -0.0117 | +0.0057 | +0.0042 | -0.0013 | -0.0265 | -0.0092 |

## Effect of calibration set: UltraChat vs perfect-blend

All 6 pairs. Calibration batch only &mdash; every observer-batch checkpoint is calibrated on perfect-blend, so this axis cannot be widened to 20. **recovery(ultrachat) &minus; recovery(perfect-blend)**; bold = exceeds that task's noise floor.

| Held fixed | gsm8k | mmlu_pro | ifeval | math_500 | gpqa | aime25 | Mean |
|---|---|---|---|---|---|---|---|
| AWQ / perfect-blend | +0.0110 | **+0.0381** | -0.0018 | **-0.0328** | +0.0026 | -0.0588 | -0.0070 |
| GPTQ / perfect-blend | -0.0078 | **+0.0053** | +0.0076 | **-0.0189** | +0.0283 | -0.0118 | +0.0005 |
| GPTQ+AWQ / perfect-blend | **+0.0145** | **+0.0549** | +0.0031 | +0.0015 | -0.0103 | +0.0353 | +0.0165 |
| GPTQ+SmoothQuant / perfect-blend | -0.0032 | **+0.0152** | -0.0067 | **-0.0292** | **+0.0541** | +0.0353 | +0.0109 |
| SmoothQuant / perfect-blend | **+0.0319** | **+0.0222** | -0.0076 | +0.0022 | +0.0103 | +0.0294 | +0.0147 |
| none / perfect-blend | -0.0058 | +0.0030 | -0.0054 | -0.0117 | -0.0052 | -0.0529 | -0.0130 |
| **Mean** (6 pairs) | +0.0068 | +0.0231 | -0.0018 | -0.0148 | +0.0133 | -0.0039 | +0.0038 |

## Effect of the weight observer: mse vs imatrix &dagger;

All 4 pairs. Observer batch only &mdash; every calibration-batch checkpoint uses the default observer, so this axis cannot be widened to 20. **recovery(mse) &minus; recovery(imatrix)**; bold = exceeds that task's noise floor.

| Held fixed | gsm8k | mmlu_pro | ifeval | math_500 | gpqa | aime25 | Mean |
|---|---|---|---|---|---|---|---|
| plain | **-0.0229** | **-0.0275** | -0.0054 | **+0.0510** | **-0.0515** | -0.0294 | -0.0143 |
| AWQ | -0.0052 | **+0.0041** | -0.0049 | +0.0066 | +0.0026 | +0.0176 | +0.0035 |
| GPTQ | **+0.0122** | +0.0028 | +0.0058 | **-0.0277** | -0.0284 | -0.0235 | -0.0098 |
| GPTQ+AWQ | **+0.0243** | **+0.0083** | -0.0076 | **+0.0189** | -0.0232 | -0.0647 | -0.0073 |
| **Mean** (4 pairs) | +0.0021 | -0.0031 | -0.0030 | +0.0122 | -0.0251 | -0.0250 | -0.0070 |

## Algorithm families

All 20 variants grouped by weight-side algorithm, ignoring calibration set and observer. *Envelope* = best member on that task (so wider families get a higher envelope); *Best member* = highest 6-task mean of any single member; *Spread* = range of member 6-task means, i.e. how much the calibration/observer choice moves that family.

| Recipe | n | gsm8k | mmlu_pro | ifeval | math_500 | gpqa | aime25 | Envelope mean | Best member | Spread |
|---|---|---|---|---|---|---|---|---|---|---|
| GPTQ+AWQ | 4 | 1.0066 | 0.9986 | 0.9977 | 1.0131 | 0.9639 | 0.9883 | 0.9947 | 0.9885 (`NVFP4-GPTQ-AWQ-ultrachat`) | 0.0196 |
| GPTQ | 4 | 1.0005 | 0.9772 | 0.9951 | 1.0138 | 1.0077 | 0.9706 | 0.9942 | 0.9874 (`NVFP4-GPTQ-IMATRIX`) | 0.0098 |
| AWQ | 4 | 0.9863 | 0.9982 | 0.9969 | 1.0262 | 0.9485 | 1.0000 | 0.9927 | 0.9831 (`NVFP4-AWQ-MSE`) | 0.0099 |
| RTN | 4 | 0.9852 | 0.9996 | 0.9910 | 1.0175 | 0.9691 | 0.9647 | 0.9878 | 0.9770 (`NVFP4`) | 0.0178 |
| GPTQ+SmoothQuant | 2 | 0.9912 | 0.9828 | 1.0049 | 1.0058 | 0.9588 | 0.9412 | 0.9808 | 0.9743 (`NVFP4-GPTQ-SmoothQuant-ultrachat`) | 0.0109 |
| SmoothQuant | 2 | 0.9545 | 0.9852 | 0.9933 | 1.0182 | 0.9510 | 0.9412 | 0.9739 | 0.9726 (`NVFP4-SmoothQuant-ultrachat`) | 0.0147 |

## Run provenance

Observer batch (&dagger;), launched in two concurrent groups of four, one GPU each.
The 12 calibration & algorithm runs predate these and their run directories are
no longer retained locally; their per-seed numbers are carried forward verbatim
from the earlier revision of this document.

| Model | Run dir | Wall clock | Items | Failures |
|-------|---------|------------|-------|----------|
| NVFP4-IMATRIX | `runs/20260920_233539_qwen3-30b-a3b-nvfp4_full` | 9.42 h | 23/23 | 0 |
| NVFP4-MSE | `runs/20260920_234123_qwen3-30b-a3b-nvfp4-mse_full` | 9.19 h | 23/23 | 0 |
| NVFP4-GPTQ-IMATRIX | `runs/20260920_235448_gptq_imatrix_full` | 9.35 h | 23/23 | 0 |
| NVFP4-GPTQ-MSE | `runs/20260920_235453_gptq_mse_full` | 9.51 h | 23/23 | 0 |
| NVFP4-AWQ-IMATRIX | `runs/20260921_131011_awq_imatrix_full` | 8.90 h | 23/23 | 0 |
| NVFP4-AWQ-MSE | `runs/20260921_131016_awq_mse_full` | 8.95 h | 23/23 | 0 |
| NVFP4-GPTQ-AWQ-IMATRIX | `runs/20260921_131021_gptq_awq_imatrix_full` | 8.64 h | 23/23 | 0 |
| NVFP4-GPTQ-AWQ-MSE | `runs/20260921_131026_gptq_awq_mse_full` | 8.55 h | 23/23 | 0 |

Regenerate the raw per-seed comparison for the observer batch with `python3
/home/dsikka/compare_runs.py`. The merged 20-model recovery, aggregate and
per-task tables above are produced by `python3 /home/dsikka/merge_tables.py`,
which recomputes the observer batch from its run directories and re-parses the
12 older variants' per-seed numbers out of this file — so it must be run
against a revision that still contains them. The observer-batch delta tables
come from `python3 /home/dsikka/gen_obs_tables.py`.

### Reproduction notes

- Runs were launched via `/home/dsikka/run_full_generic.sh <MODEL_PATH>
  <SERVED_NAME> <PORT> <TAG>`, which hardcodes every eval-relevant setting so
  concurrent runs cannot drift apart.
- **Concurrent runs need cache isolation.** `runner._clear_lighteval_caches()`
  rmtree's the CWD-relative `.litellm_cache` *and* the entire `$HF_HOME/lighteval`
  tree before each lighteval task, so parallel runs sharing either will clobber
  each other mid-generation. Each run gets its own CWD (`wd_<tag>`) and
  `HF_HOME` (`.cache/hf_<tag>`), with hub/datasets/xet caches shared. Copy
  `token` and `stored_tokens` into each new `HF_HOME` or gated GPQA Diamond 401s.
- lm-eval ignores the `--output_path` filename and appends its own ISO
  timestamp. `eval_agent/summary.py` was fixed to glob for the real file;
  before that fix, every lm-eval score was silently dropped from
  `summary_data.json`.

## Source

Ported from [issue #3001](https://github.com/vllm-project/llm-compressor/issues/3001),
where this sweep was originally published. The reproduction scripts referenced
under run provenance are local to the author's machine and are not published.
