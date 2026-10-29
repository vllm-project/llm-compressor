# W4A16 iMatrix Ablation: Full Results Summary

## Executive summary

This sweep compares 48 quantized configurations across four models, six error powers, and two iMatrix objectives. Each configuration has three seeds for each of four evaluations: MMLU, GSM8K Platinum, GPQA Diamond, and Math500. That gives 576 configuration/benchmark/seed outcomes. All 48 configurations completed; the saved status records zero final task failures for the original three-benchmark evaluation and all three MMLU seed passes.

No single objective or error power wins across all benchmarks. Averaged evenly over the 48 configurations, Group-absmax changes the scores only slightly versus the base objective: +0.06 percentage points on GSM8K strict, +0.12 on GSM8K flexible, +0.23 on GPQA Diamond, −0.18 on Math500, and −0.02 on MMLU. The best across-model power depends on the benchmark: p=2 for MMLU and GSM8K, p=2.8 for GPQA Diamond, and p=2.6 for Math500.

**MMLU caveat:** all 36 Qwen3.8 MMLU logs (12 configurations × 3 seeds) contain null-content warnings. Together these are 28,641 warnings across 505,512 requests (about 5.7%). The evaluator says these responses were replaced by a no-answer placeholder and advises checking the reasoning-content field or generation limit. Treat Qwen3.8 MMLU scores with that caveat; the runs completed successfully, but the warnings may affect its measured score.

## Experiment design and evaluation

- **Quantization:** W4A16, group size 64, `imatrix_mse` observer, Triton backend, patience 0, grid 100, maxshrink 0.95.
- **Calibration:** 512 PerfectBlend training examples, sequence length 2048, calibration seed 42.
- **Models:** Llama 3.1 8B Instruct; Qwen3.8 27B; Qwen3 30B A3B Instruct 2507; Nemotron 3.5 Lightning 30B A3B BF16.
- **Error powers:** 2.0, 2.2, 2.4, 2.6, 2.8, and 3.0.
- **Objectives:** Base = `importance × |w − q(w)|^p`; Group-absmax = `importance × |w − q(w)|^p × sqrt(w² + group_absmax²)`.
- **Evaluation seeds:** 1234, 2345, and 3456, run in global seed passes.
- **Evaluation metrics:** MMLU CoT strict exact match; GSM8K Platinum strict-match and flexible-extract; GPQA Diamond pass@1; Math500 pass@1. Every reported score is the mean across three seeds.
- **Generation caps:** GSM8K 8,192 tokens; MMLU 8,192; GPQA Diamond 65,536; Math500 16,384. MMLU used 5-shot prompts with chat templates and few-shot examples as multi-turn messages.
- **Base sampling settings in the experiment manifest:** Llama 3.1 uses temperature 0.6 / top-p 0.9; Qwen3.8 uses 1.0 / 0.95 / top-k 20; Qwen3 30B uses 0.7 / 0.8 / top-k 20; Nemotron uses 1.0 / 0.95. Task runners may add task-specific generation settings; the per-run manifests and commands are the source of record.

## Best individual configurations

| Metric | Best score | Model | Objective | Power |
|---|---:|---|---|---:|
| MMLU | 86.50% | Qwen3.8 27B | Group-absmax | 2 |
| GSM8K strict | 97.99% | Qwen3 30B A3B Instruct 2507 | Base | 2.4 |
| GSM8K flexible | 97.99% | Qwen3 30B A3B Instruct 2507 | Base | 2.4 |
| GPQA Diamond | 90.57% | Qwen3.8 27B | Group-absmax | 2.2 |
| Math500 | 88.73% | Qwen3 30B A3B Instruct 2507 | Base | 3 |

GSM8K Platinum flexible-extract is tied at 97.99%: the Qwen3 30B A3B base objective at p=2.4 and Nemotron 3.5 Lightning Group-absmax at p=2.2.

## Mean score by model

Each cell is the arithmetic mean across the model’s 12 objective/power configurations; these model means are descriptive and should not be interpreted as a calibrated comparison against BF16 baselines.

| Model | MMLU | GSM8K strict | GSM8K flexible | GPQA Diamond | Math500 |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B Instruct | 69.65% | 82.89% | 83.09% | 29.56% | 42.99% |
| Qwen3.8 27B | 86.27% | 96.89% | 97.03% | 89.67% | 84.38% |
| Qwen3 30B A3B Instruct 2507 | 82.86% | 97.74% | 97.75% | 54.12% | 86.62% |
| Nemotron 3.5 Lightning 30B A3B BF16 | 81.48% | 97.32% | 97.75% | 60.52% | 82.91% |

## Objective effect averaged across the sweep

Averages give equal weight to all 24 model/power settings per objective. Delta is Group-absmax minus Base, in percentage points.

| Metric | Base | Group-absmax | Delta (pp) |
|---|---:|---:|---:|
| MMLU | 80.07% | 80.05% | -0.02 |
| GSM8K strict | 93.68% | 93.74% | +0.06 |
| GSM8K flexible | 93.84% | 93.96% | +0.12 |
| GPQA Diamond | 58.35% | 58.59% | +0.23 |
| Math500 | 74.31% | 74.13% | -0.18 |

## Mean score by error power

Each mean combines the four models and both objectives (eight configurations per power).

| Power | MMLU | GSM8K strict | GSM8K flexible | GPQA Diamond | Math500 |
|---:|---:|---:|---:|---:|---:|
| 2 | 80.47% | 93.84% | 94.02% | 58.00% | 73.92% |
| 2.2 | 80.12% | 93.69% | 93.89% | 58.02% | 73.99% |
| 2.4 | 80.18% | 93.69% | 93.83% | 58.65% | 73.92% |
| 2.6 | 80.05% | 93.62% | 93.80% | 58.65% | 74.67% |
| 2.8 | 79.83% | 93.81% | 94.00% | 59.15% | 74.44% |
| 3 | 79.72% | 93.63% | 93.88% | 58.35% | 74.41% |

## Complete per-configuration results

Scores are means over seeds. For MMLU, seed-level values are retained in the linked MMLU matrix. GPQA Diamond and Math500 standard errors are retained in the original results matrix.

### Llama 3.1 8B Instruct

| Objective | Power | MMLU | GSM8K strict | GSM8K flexible | GPQA Diamond | Math500 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 2 | 69.53% | 83.32% | 83.48% | 27.10% | 43.47% |
| Group-absmax | 2 | 69.95% | 83.54% | 83.82% | 30.30% | 42.80% |
| Base | 2.2 | 69.57% | 83.24% | 83.51% | 28.79% | 41.80% |
| Group-absmax | 2.2 | 69.63% | 82.80% | 83.04% | 30.81% | 43.13% |
| Base | 2.4 | 69.74% | 82.69% | 82.88% | 30.81% | 42.67% |
| Group-absmax | 2.4 | 69.45% | 82.63% | 82.96% | 27.61% | 41.93% |
| Base | 2.6 | 69.69% | 82.44% | 82.63% | 30.13% | 44.80% |
| Group-absmax | 2.6 | 69.69% | 82.13% | 82.27% | 29.63% | 43.73% |
| Base | 2.8 | 69.48% | 82.96% | 83.07% | 29.80% | 42.47% |
| Group-absmax | 2.8 | 69.77% | 83.40% | 83.51% | 31.99% | 42.93% |
| Base | 3 | 69.66% | 82.49% | 82.63% | 29.63% | 43.60% |
| Group-absmax | 3 | 69.58% | 83.10% | 83.24% | 28.11% | 42.53% |

### Qwen3.8 27B

| Objective | Power | MMLU | GSM8K strict | GSM8K flexible | GPQA Diamond | Math500 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 2 | 86.29% | 96.94% | 97.02% | 90.40% | 83.93% |
| Group-absmax | 2 | 86.50% | 96.86% | 97.38% | 89.90% | 84.20% |
| Base | 2.2 | 86.38% | 96.50% | 96.66% | 88.89% | 84.00% |
| Group-absmax | 2.2 | 86.43% | 96.91% | 97.02% | 90.57% | 84.93% |
| Base | 2.4 | 86.42% | 96.80% | 96.91% | 88.72% | 84.73% |
| Group-absmax | 2.4 | 86.49% | 96.94% | 96.97% | 90.24% | 85.07% |
| Base | 2.6 | 86.38% | 97.22% | 97.27% | 89.90% | 84.20% |
| Group-absmax | 2.6 | 86.41% | 97.13% | 97.27% | 89.56% | 84.13% |
| Base | 2.8 | 86.13% | 96.97% | 96.99% | 90.07% | 84.13% |
| Group-absmax | 2.8 | 86.05% | 97.11% | 97.22% | 90.24% | 84.13% |
| Base | 3 | 85.85% | 96.83% | 96.88% | 89.39% | 84.47% |
| Group-absmax | 3 | 85.91% | 96.53% | 96.72% | 88.22% | 84.60% |

### Qwen3 30B A3B Instruct 2507

| Objective | Power | MMLU | GSM8K strict | GSM8K flexible | GPQA Diamond | Math500 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 2 | 82.67% | 97.33% | 97.38% | 53.03% | 85.60% |
| Group-absmax | 2 | 82.70% | 97.90% | 97.90% | 52.36% | 84.87% |
| Base | 2.2 | 82.62% | 97.52% | 97.52% | 52.19% | 86.80% |
| Group-absmax | 2.2 | 82.40% | 97.63% | 97.66% | 51.85% | 87.13% |
| Base | 2.4 | 82.89% | 97.99% | 97.99% | 55.56% | 84.67% |
| Group-absmax | 2.4 | 83.01% | 97.77% | 97.77% | 53.54% | 87.47% |
| Base | 2.6 | 82.93% | 97.77% | 97.77% | 55.72% | 87.73% |
| Group-absmax | 2.6 | 83.14% | 97.66% | 97.66% | 54.55% | 86.87% |
| Base | 2.8 | 82.79% | 97.74% | 97.74% | 54.88% | 87.67% |
| Group-absmax | 2.8 | 83.07% | 97.77% | 97.77% | 56.23% | 87.00% |
| Base | 3 | 82.98% | 97.90% | 97.90% | 54.88% | 88.73% |
| Group-absmax | 3 | 83.11% | 97.96% | 97.96% | 54.71% | 84.93% |

### Nemotron 3.5 Lightning 30B A3B BF16

| Objective | Power | MMLU | GSM8K strict | GSM8K flexible | GPQA Diamond | Math500 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 2 | 83.63% | 97.33% | 97.52% | 60.94% | 83.60% |
| Group-absmax | 2 | 82.51% | 97.49% | 97.63% | 59.93% | 82.87% |
| Base | 2.2 | 82.26% | 97.49% | 97.71% | 59.43% | 82.20% |
| Group-absmax | 2.2 | 81.66% | 97.46% | 97.99% | 61.62% | 81.93% |
| Base | 2.4 | 81.93% | 96.99% | 97.24% | 60.10% | 82.87% |
| Group-absmax | 2.4 | 81.54% | 97.71% | 97.93% | 62.63% | 81.93% |
| Base | 2.6 | 81.39% | 97.13% | 97.77% | 59.43% | 82.33% |
| Group-absmax | 2.6 | 80.79% | 97.52% | 97.77% | 60.27% | 83.53% |
| Base | 2.8 | 80.94% | 97.38% | 97.82% | 58.59% | 83.80% |
| Group-absmax | 2.8 | 80.42% | 97.13% | 97.88% | 61.45% | 83.40% |
| Base | 3 | 79.64% | 97.46% | 97.96% | 62.12% | 83.27% |
| Group-absmax | 3 | 81.06% | 96.77% | 97.74% | 59.76% | 83.13% |

## MMLU seed-level summary

Mean strict exact-match score across all 48 configurations for each seed:

| Seed | Mean score |
|---:|---:|
| 1234 | 80.00% |
| 2345 | 80.13% |
| 3456 | 80.07% |

## Interpretation and limitations

- The group-absmax factor has a small positive aggregate effect on GSM8K and GPQA, and a small negative aggregate effect on Math500 and MMLU. Its direction varies by model; for example, it improves GPQA means for Nemotron but lowers GPQA means for Qwen3-30B-A3B.
- Across powers, p=2 gives the best aggregate MMLU and GSM8K means; higher powers do not improve those averages. GPQA’s best aggregate is p=2.8, while Math500 peaks at p=2.6.
- These are comparisons among W4A16 ablations. There is no unquantized reference score in this result set, so the sweep alone does not quantify degradation from BF16 or establish absolute quality retention.
- Scores use three stochastic evaluation seeds. Small differences—especially a few tenths of a percentage point—should not be treated as conclusive without additional seeds or paired significance analysis.
- The Qwen3.8 MMLU null-content warnings are a specific data-quality concern. All other models’ MMLU logs had zero such warnings.

## Source artifacts

- [Experiment manifest](/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02/manifest.json)
- [Original three-benchmark summary](/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02/RESULTS.md)
- [GSM8K Platinum, GPQA Diamond, and Math500 results matrix CSV](/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02/results_matrix.csv)
- [MMLU results matrix CSV, including seed scores](/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02/mmlu_results_matrix.csv)
- [MMLU orchestration status](/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02/mmlu_status.json)
- [Quantization and evaluation artifacts](/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02/evaluations/)
