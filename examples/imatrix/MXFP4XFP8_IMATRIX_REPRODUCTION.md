# Reproducing the MXFP4xFP8 imatrix ablation

This runner reproduces the MXFP4 weight plus dynamic FP8 activation imatrix
ablation. It uses the `MXFP4XFP8` compressed-tensors preset and evaluates the
same three-model matrix as the completed run. The Qwen2.5-7B-Instruct arm
replaces the cancelled Llama 3.1-8B arm. Nemotron is not part of the completed
evaluation matrix. Result tables are not included in this reproduction guide.

The observer implementation is in this repository and was introduced in
[PR #3348](https://github.com/vllm-project/llm-compressor/pull/3348). The
quantization preset is provided by
[compressed-tensors PR #913](https://github.com/vllm-project/compressed-tensors/pull/913).
The completed run used llm-compressor commit
`6870c15904c8e8184a191cb35c8c03979cb7dda1`, compressed-tensors commit
`c4998704628c30f07c9959f98c3b3b4e481c8026`, and the pinned model and dataset
revisions below. The runner writes these revisions and source hashes to
`experiment_manifest.json`.

## Environment

Follow the setup in the `eval-agent` skill's `SKILL.md`, including its base,
lm-eval, and lighteval environments. Evaluation uses `eval-agent` with vLLM;
quantization runs in `rhdev`. Install this repository in the `rhdev` environment
in editable mode. Install compressed-tensors at the pinned revision in that
environment so the `MXFP4XFP8` preset matches the completed run.

For example, after checking out this llm-compressor branch:

```bash
export RHVENV_PYTHON=/path/to/rhdev/bin/python
git clone https://github.com/vllm-project/compressed-tensors.git
git -C compressed-tensors checkout c4998704628c30f07c9959f98c3b3b4e481c8026
"$RHVENV_PYTHON" -m pip install -e .
"$RHVENV_PYTHON" -m pip install -e ./compressed-tensors
```

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

The recorded environment used llm-compressor package metadata
`0.14.1.dev24+g5f1b9c2ba`, compressed-tensors
`0.19.1.dev14+gc4998704`, PyTorch `2.13.0+cu130`, Transformers `5.17.0`, and
Datasets `4.8.5` for quantization. vLLM was
`0.33.0rc1.dev277+g4440f0e26` with PyTorch `2.14.0+cu130` for evaluation. The
eval-agent registry must match SHA-256
`b0d21efbde41ad245ce38a087573f6bf6cd527852b782b6e01a6718cf3006bac`; the
registry defines the task prompts, scoring, repetitions, and seed behavior.

## Ablation definition

- Scheme: the `MXFP4XFP8` preset, targeting `Linear` modules. Weights use
  symmetric floating-point group quantization with 4 bits, group size 32, and
  E8M0 scales. Activations use symmetric dynamic 8-bit floating-point
  quantization with group size 128.
- Calibration: `mlabonne/open-perfectblend`, revision
  `af60f3c18201652a83a93f46fcfee1b646ba3df7`, split `train[:512]`, 512 samples,
  maximum sequence length 2048, and BF16 calibration precision.
- Models and pinned revisions:
  - `Qwen/Qwen2.5-7B-Instruct` —
    `a09a35458c702b33eeacc393d103063234e8bc28`
  - `Qwen/Qwen3.8-27B` — `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  - `Qwen/Qwen3-30B-A3B` — `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Powers: 2.0, 2.2, 2.4, 2.6, 2.8, and 3.0.
- Objectives: `importance * abs(QDQ(w) - w)^power`, and the same score times
  `sqrt(w^2 + group_absmax^2)` (`magnitude_normalization=True`).
- Patience: 0, so the search evaluates its full candidate list.
- Seeds: 1234, 2345, and 3456, evaluated seed-major.
- Observer variants: Qwen3.8 runs both default and expanded settings; Qwen2.5
  and Qwen3 use expanded settings.
- Matrix size: 48 quantization configurations across the three models.

Both observer variants explicitly set `grid_type="exponential"`. This matches
the completed run, where `grid_type=None` auto-selected exponential spacing for
E8M0 scales. The default observer (`imatrix_mse`) uses grid 100, `expand=1.0`,
and `maxshrink=0.95`; its candidates include powers of two through 1/16 and the
legacy terminal point 0.06. The expanded observer
(`nvfp4_expanded_imatrix`) uses grid 200, `expand=1.0`, and
`maxshrink=63/64`; its candidates include powers of two through 1/64 and the
legacy terminal point 0.025. The terminal point preserves the final candidate
of the corresponding linear grid.

On the recorded H100/SM90 evaluation setup, vLLM selected the
`MarlinMxFp4LinearKernel` and `MarlinExperts` weight-only kernels. Those kernels
ignored the checkpoint's FP8 activation quantization parameters, so evaluation
measured MXFP4 weights with BF16 activations (W4A16 behavior). The checkpoint
itself still contains the MXFP4xFP8 preset configuration. The runner records
this distinction in the manifest; a vLLM backend that consumes FP8 activations
would evaluate a different effective scheme.

## Run the ablation

The runner uses `chg run` for quantization and evaluation GPU reservations. The
worker count limits concurrent jobs; six workers matches the completed run's
GPU cap. Run smoke checks first, then the complete seed-major matrix:

```bash
SCRIPT=examples/imatrix/run_mxfp4xfp8_imatrix_ablation.py
RUN_ROOT="$PWD/runs/mxfp4xfp8_imatrix_ablation"

python "$SCRIPT" prepare --root "$RUN_ROOT"
python "$SCRIPT" smoke --root "$RUN_ROOT" --workers 4
python "$SCRIPT" full --root "$RUN_ROOT" --workers 6 \
  --seeds 1234,2345,3456
```

Smoke evaluation omits MMLU because the registry's MMLU smoke workload is very
large. The full run evaluates `gsm8k_platinum_cot_llama`, `mmlu_cot_llama`,
`math_500`, and `gpqa_diamond`. The runner records checkpoints, per-task logs,
progress, GPU utilization, and `chg status` under the run root. It retries each
evaluation at most once after the initial attempt and preserves completed
seed rounds if interrupted.

The runner writes summaries inside the chosen run root. Keep those generated
artifacts separate from source changes; this PR contains reproduction code and
settings only.
