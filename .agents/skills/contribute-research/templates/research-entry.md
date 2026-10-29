# Research entry template

Read [the research index](../../../../docs/research/index.md) first; it remains
the source of truth. Copy the Markdown block, replace placeholders, and remove
unused optional sections and columns. Use the index's examples for multiple
authors and updated dates.

```markdown
---
title: "SHORT_DESCRIPTION"
tags:
  - ALGORITHM
  - SCHEME
  - MODEL_FAMILY
---

<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>

# SHORT_DESCRIPTION

## Summary

One or two sentences on the measured result and its scope.

## Setup

- Model(s): `MODEL_ID`, revision `REVISION`, baseline precision `PRECISION`
- Scheme/recipe: `SCHEME`, modifiers and parameters `RECIPE_SETTINGS`
- Calibration: `DATASET`, `SPLIT`, `SEQUENCE_COUNT` sequences, `SEQUENCE_LENGTH` tokens
- Hardware: `ACCELERATOR_COUNT_AND_MODEL`, `MEMORY`
- Software/serving: `VERSIONS_OR_COMMITS`, `RUNTIME_SETTINGS`
- Evaluation: `TASKS`, `METRICS`, `EVALUATION_SETTINGS`, `SEEDS_AND_REPEATS`

## Results

### Primary tasks

Name the tasks, metrics, and units. Define recovery or averages if reported.
Repeat the table per model when needed; mark unmeasured values explicitly.

| Configuration | TASK_A_METRIC | TASK_B_METRIC | Recovery | Throughput (UNIT) | Weight size (UNIT) |
|---------------|---------------|---------------|----------|-------------------|--------------------|
| Baseline | SCORE | SCORE | - | MEASURED_VALUE | MEASURED_VALUE |
| VARIANT | SCORE | SCORE | MEASURED_VALUE | MEASURED_VALUE | MEASURED_VALUE |

### Additional evaluation (optional)

Describe the additional task family and any evaluation settings that changed.

| Configuration | TASK_METRIC | Recovery |
|---------------|-------------|----------|
| Baseline | SCORE | - |
| VARIANT | SCORE | MEASURED_VALUE |

### Calibration ablations (optional)

State what varied and what stayed fixed. Identify confounded comparisons.

| Calibration dataset | Sequence count | TASK_METRIC |
|---------------------|----------------|-------------|
| DATASET_A | COUNT | SCORE |
| DATASET_B | COUNT | SCORE |

## Observations

### FINDING

Explain what the measurements support and what remains a hypothesis.
Include negative results and surprises. Repeat this subsection for other findings.

### Limitations and next steps

Describe variability, operational issues, what was not tested, and what to try next.

## Independent replication (optional)

Link the replication source and identify its contributor, setup differences,
and whether the findings agreed. Follow the index's credit and consent rules.

## Reproducing

[Scripts, recipes, and raw results](REPRODUCTION_URL)

State the commands or steps needed to reproduce the reported runs.
```
