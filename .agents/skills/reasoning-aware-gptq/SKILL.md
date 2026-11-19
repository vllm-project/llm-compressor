---
name: reasoning-aware-gptq
description: Reproduce or extend the Qwen3 GPTQ reasoning-aware calibration experiment.
---

# Reasoning-aware GPTQ experiments

Use the [`reasoning_aware_calibration` example](../../../examples/gptq/reasoning_aware_calibration/README.md)
as the source of truth for environments, commands, sample IDs, and measured findings.

1. Keep generation, GPTQ calibration, and evaluation in their documented environments. Put generated checkpoints
   and logs outside the repository.
2. Regenerate traces and calibrate with `run_experiment.sh`. The three arms must retain the same 500 calibration
   questions and active Hessian-token budget. Do not change the pinned dataset revision or prompt/evaluation ID files
   when reproducing the reported run.
3. Run evaluations with `run_evaluations.sh` or the manual-server flow in the example README. Score the full model
   completion against the full reference solution using `score_math_verify.py`; native exact-match extraction is
   retained only as a parser audit and is not the reported accuracy metric.
4. Use `compare_output_tokens.py` with the per-sample Math-Verify sidecars to compare answer-token counts against
   BF16. Its bottom-k metric is the arithmetic mean of the k smallest per-question ratios; its P90 is nearest-rank.
5. Report model, scheme, calibration token count, hardware, evaluation IDs, generation settings, Math-Verify version,
   and any sample/scorer errors. State clearly when a checkpoint was quantized but not evaluated. Do not present the
   existing single-run intervals as covering generation-run variance.
6. When updating results, preserve the research-entry conventions in
   [`contribute-research`](../contribute-research/SKILL.md) and update the Open Research Hub entry.
