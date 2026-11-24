# Reproducing the MXFP4A16 imatrix ablation

This runner reproduces the MXFP4 weight-only, BF16-activation ablation and its
Qwen3.8 grid-range comparisons. It uses the imatrix observer only; it does not
run the separate MSE-only or MXFP4xFP8 ablations.

The observer implementation is in this repository and was introduced in
[PR #3348](https://github.com/vllm-project/llm-compressor/pull/3348). The
original quantization run used repository commit
`5f1b9c2ba78ce40d618f2b677d8e075c382b79f1`; the experiment runner pins the
same model and calibration dataset revisions recorded in those runs.

## Environment

Follow the setup in the `eval-agent` skill's `SKILL.md`, including its base,
lm-eval, and lighteval environments. This runner uses the eval-agent CLI for
evaluation, the configured vLLM environment for serving, and the `rhdev`
environment for quantization. Install this repository into `rhdev` in editable
mode so quantization imports the checked out observer code.

Set these paths for your machine before running:

```bash
export EVAL_AGENT_SKILL_DIR=/path/to/evaluation/eval-agent
export EVAL_AGENT=/path/to/llm-compressor/.venv/bin/eval-agent
export VLLM_ENV=/path/to/vllm
export VLLM_BINARY="$VLLM_ENV/bin/vllm"
export LIGHTEVAL_ENV=/path/to/llm-compressor/.venvs/lighteval
export RHVENV_PYTHON=/path/to/rhdev/bin/python
export HF_HOME=/path/to/large/huggingface/cache
```

The `eval-agent` registry must match SHA-256
`b0d21efbde41ad245ce38a087573f6bf6cd527852b782b6e01a6718cf3006bac` for the
same task definitions and scores. The pinned model revisions and calibration
dataset revision are written into `experiment_manifest.json` by `prepare`.

The completed evaluation environment used vLLM
`0.33.0rc1.dev277+g4440f0e26`, PyTorch `2.13.0+cu130`, and the Marlin MXFP4
weight-only kernels (`MarlinMxFp4LinearKernel` for Linear layers and
`MarlinExperts` for MoE experts) on H100/SM90 GPUs.

## Ablation definition

- Scheme: MXFP4 weights with BF16 activations (`input_activations=None`),
  symmetric floating-point group quantization, group size 32, and E8M0 scales
  (`scale_dtype=torch.uint8`).
- Calibration: `mlabonne/open-perfectblend`, pinned revision
  `af60f3c18201652a83a93f46fcfee1b646ba3df7`, `train[:512]`, 512 samples, and
  maximum sequence length 2048.
- Powers: 2.0, 2.2, 2.4, 2.6, 2.8, and 3.0.
- Objectives: `importance * abs(QDQ(w) - w)^power`, and the same score times
  `sqrt(w^2 + group_absmax^2)` (enabled by `magnitude_normalization=True`).
- Patience: 0.
- The runner explicitly sets `grid_type="linear"` to preserve the search used
  for the reported results. Current MX scale observers otherwise default to an
  exponential grid. The historical linear loop uses indices
  `0 .. int(grid * maxshrink) - 1`; the final point is one grid step above the
  nominal `maxshrink` boundary. The actual last multipliers are 0.06 for the
  default grid, 0.81 for the expanded grid, 0.05 for grid 180 full-range, and
  0.045 for grid 200 full-range.
- Observer settings: default is `imatrix_mse`, grid 100, expand 1.0,
  maxshrink 0.95; expanded is `nvfp4_expanded_imatrix`, grid 200, expand 1.8,
  maxshrink `1 - 0.8/1.8`. The Qwen3.8 comparison arms use grid 180 or 200,
  expand 1.8, and maxshrink 0.98.
- Evaluations use the immutable eval-agent tasks `gsm8k_platinum_cot_llama`,
  `mmlu_cot_llama`, `math_500`, and `gpqa_diamond`. Smoke runs omit MMLU; the
  full run includes it. `eval-agent` and the skill registry define prompts,
  scoring, repetitions, and seeds.

## Run the primary ablation

The original three-model run covered Qwen3.8-27B, Qwen3-30B-A3B, and
Nemotron 3.5 Lightning-30B-A3B for seeds 1234 and 2345. Qwen3.8 has both
default and expanded observer settings; the other two use expanded settings.
Set separate roots for the original run and the Qwen2.5 replacement so their
manifests and progress files remain independent.

```bash
SCRIPT=examples/imatrix/run_mxfp4a16_imatrix_ablation.py

# Original three-model matrix (48 quantization configurations).
export MXFP4_ABLATION_EXCLUDE_MODELS=qwen25_7b
export MXFP4A16_RUN_ROOT="$PWD/runs/mxfp4a16_original_models"
python "$SCRIPT" prepare --root "$MXFP4A16_RUN_ROOT"
python "$SCRIPT" smoke --root "$MXFP4A16_RUN_ROOT" --workers 4
python "$SCRIPT" full --root "$MXFP4A16_RUN_ROOT" --workers 8 \
  --seeds 1234,2345

# Qwen2.5-7B-Instruct replaced the cancelled Llama 3.1-8B arm; it used 3 seeds.
export MXFP4_ABLATION_EXCLUDE_MODELS=qwen38_27b,qwen3_30b_a3b,nemotron35_30b_a3b
export MXFP4A16_RUN_ROOT="$PWD/runs/mxfp4a16_qwen25_replacement"
python "$SCRIPT" prepare --root "$MXFP4A16_RUN_ROOT"
python "$SCRIPT" smoke --root "$MXFP4A16_RUN_ROOT" --workers 4
python "$SCRIPT" full --root "$MXFP4A16_RUN_ROOT" --workers 8 \
  --seeds 1234,2345,3456
```

Each quantization and vLLM process is launched through `chg run` so it obtains a
GPU reservation from canhazgpu. The worker count caps concurrent jobs; lower it
if fewer GPUs are available. The runner records GPU utilization and `chg status`
every ten minutes, writes per-job logs and manifests, and runs seed-major so an
interrupted run retains complete earlier seed rounds. `full` requires passing
smoke checks first.

## Reproduce the Qwen3.8 scale-range comparisons

Run these after the original three-model matrix has completed. The comparison
commands reuse its baseline checkpoint and use seeds 1234 and 2345 for grid 180,
and 1234, 2345, and 3456 for grid 200.

```bash
export MXFP4_ABLATION_EXCLUDE_MODELS=qwen25_7b
export MXFP4A16_RUN_ROOT="$PWD/runs/mxfp4a16_original_models"
python examples/imatrix/run_mxfp4a16_imatrix_ablation.py compare-grid180 \
  --baseline-root "$MXFP4A16_RUN_ROOT" \
  --root "$PWD/runs/mxfp4a16_qwen38_grid180" \
  --workers 8 --seeds 1234,2345
python examples/imatrix/run_mxfp4a16_imatrix_ablation.py compare-grid200-fullrange \
  --baseline-root "$MXFP4A16_RUN_ROOT" \
  --root "$PWD/runs/mxfp4a16_qwen38_grid200_fullrange" \
  --workers 8 --seeds 1234,2345,3456
```

Use each run's `ablation_summary.md`, `experiment_manifest.json`, and
`evaluations/` artifacts alongside the aggregate result table in the companion
gist. The manifest records model/data revisions, observer settings, evaluation
registry hash, and source hashes.
