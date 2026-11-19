---
name: mxfp4-ablation-reproduction
description: >-
  Reproduce MXFP4A16 and MXFP4xFP8 imatrix, GPTQ plus imatrix, and model-free
  MSE ablations using the pinned runners and evaluation setup.
---

# Reproduce the MXFP4 ablations

Use this skill when rerunning the MXFP4 observer ablations or checking that a
run matches the recorded methodology. Read
[`examples/imatrix/MXFP4_ABLATION_REPRODUCTION.md`](../../../examples/imatrix/MXFP4_ABLATION_REPRODUCTION.md)
first; it links to each method's detailed settings and commands. Do not use
result tables as configuration input. The runners create per-run manifests
that record the effective configuration and applicable pinned model, dataset,
source, and evaluation-registry revisions.

## Environment

1. Follow the external `eval-agent` skill's `SKILL.md` to set up the evaluator,
   vLLM, lm-eval, and lighteval environments.
2. Use the `rhdev` Python environment for quantization and the configured vLLM
   environment for serving. Install this checkout and the compressed-tensors
   revision specified in the method guide into `rhdev`.
3. Set `EVAL_AGENT_SKILL_DIR`, `EVAL_AGENT`, `VLLM_ENV`, `VLLM_BINARY`,
   `LIGHTEVAL_ENV`, `RHVENV_PYTHON`, and `HF_HOME` for the host.
4. Make sure `chg` is available. The scripts reserve GPUs through `chg`; keep
   `--workers` at six or fewer on a six-GPU allocation.

## Choose a runner

- **MXFP4A16 imatrix:**
  `examples/imatrix/run_mxfp4a16_imatrix_ablation.py`. Its detailed guide is
  `examples/imatrix/MXFP4A16_IMATRIX_REPRODUCTION.md`; this runner was merged
  in llm-compressor PR #3477.
- **MXFP4xFP8 imatrix:**
  `examples/imatrix/run_mxfp4xfp8_imatrix_ablation.py`.
- **MXFP4xFP8 GPTQ plus imatrix:**
  `examples/imatrix/run_mxfp4xfp8_gptq_imatrix_ablation.py`.
- **Model-free MSE:**
  `examples/imatrix/run_mxfp4_mse_model_free_ablation.py`. Set
  `MXFP4_ABLATION_SCHEME=MXFP4A16` or `MXFP4XFP8` before starting each run.

The detailed xFP8 settings, pinned revisions, candidate scales, and command
examples are in `examples/imatrix/MXFP4XFP8_RESEARCH_REPRODUCTION.md`.

## Reproduction steps

For the imatrix and GPTQ+imatrix runners, run `prepare`, then `smoke`, then
`full`. Smoke validates checkpoint metadata, serving kernels, evaluation
configuration, and concurrency before the full seed-major evaluation.

For each MSE scheme, run `check-config`, `prepare`, and `run`. The MSE runner
performs model-free PTQ, checks the compressed checkpoint schema, runs smoke
evaluations, then evaluates seeds 1234, 2345, and 3456. Its exponential scale
search checks `1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64`; powers are 2.0 through 3.0
in 0.2 increments, with magnitude normalization both disabled and enabled. To
reproduce the MoE-scope MSE results, set `MXFP4_MSE_SCOPE_ABLATION=1` and use a
separate run directory.

Keep each scheme and method in a separate run directory. Generated manifests,
checkpoints, logs, per-seed evaluations, and summaries belong under that run
directory. Avoid copying generated result tables into the source tree.
