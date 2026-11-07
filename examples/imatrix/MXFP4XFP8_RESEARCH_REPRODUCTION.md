# Reproducing the MXFP4xFP8 ablations

This guide and its two runners reproduce the completed imatrix and model-free
MSE ablations for MXFP4 weights with dynamic FP8 activations. Results tables
are intentionally not included; each run writes its manifests, logs, status,
checkpoints, and generated summary under its run directory.

The quantization scheme follows the `MXFP4XFP8` preset from
[compressed-tensors PR #913](https://github.com/vllm-project/compressed-tensors/pull/913).
The exponential observer search and magnitude-normalization options are in
[llm-compressor PR #3348](https://github.com/vllm-project/llm-compressor/pull/3348).

## Environment

Use the `eval-agent` skill's `SKILL.md` to set up and validate its evaluation
environments. Evaluation is launched through `eval-agent` and vLLM; quantization
workers use the `rhdev` Python environment. Point the runners at the local
paths before running:

```bash
export EVAL_AGENT_SKILL_DIR=/path/to/evaluation/eval-agent
export EVAL_AGENT=/path/to/llm-compressor/.venv/bin/eval-agent
export VLLM_ENV=/path/to/vllm-venv
export VLLM_BINARY="$VLLM_ENV/bin/vllm"
export LIGHTEVAL_ENV=/path/to/lighteval-venv
export RHVENV_PYTHON=/path/to/rhdev/bin/python
export HF_HOME=/path/to/huggingface-cache
```

The completed runs used llm-compressor commit
`6870c15904c8e8184a191cb35c8c03979cb7dda1`, compressed-tensors commit
`c4998704628c30f07c9959f98c3b3b4e481c8026`, quantization PyTorch
`2.13.0+cu130`, Transformers `5.17.0`, and Datasets `4.8.5`. vLLM was
`0.33.0rc1.dev277+g4440f0e26` with PyTorch `2.14.0+cu130`. The eval-agent
registry used for the runs had SHA-256
`b0d21efbde41ad245ce38a087573f6bf6cd527852b782b6e01a6718cf3006bac`.
Each runner records model, dataset, source, and registry revisions in its
experiment manifest.

The `chg` command must be available for GPU reservation. Both runners use it
for GPU work. Keep the worker count at six or fewer if reproducing with the
same GPU allocation.

Install the repository and the matching compressed-tensors revision into the
`rhdev` environment used by the quantization workers:

```bash
git clone https://github.com/vllm-project/compressed-tensors.git
git -C compressed-tensors checkout c4998704628c30f07c9959f98c3b3b4e481c8026
"$RHVENV_PYTHON" -m pip install -e .
"$RHVENV_PYTHON" -m pip install -e ./compressed-tensors
```

Use the vLLM environment and evaluation dependencies validated by the
`eval-agent` skill for serving and benchmark execution.

## Shared scheme and evaluation

- The preset targets `Linear` modules. Weights use symmetric 4-bit floating
  point group quantization, group size 32, and E8M0 scales. Inputs use dynamic
  symmetric FP8 group quantization, group size 128.
- Calibration for the imatrix run uses
  `mlabonne/open-perfectblend@af60f3c18201652a83a93f46fcfee1b646ba3df7`,
  split `train[:512]`, 512 samples, maximum sequence length 2048, and BF16
  calibration precision.
- Models are pinned to these revisions:
  - `Qwen/Qwen2.5-7B-Instruct` —
    `a09a35458c702b33eeacc393d103063234e8bc28`
  - `Qwen/Qwen3.8-27B` — `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  - `Qwen/Qwen3-30B-A3B` — `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Qwen2.5-7B-Instruct replaced the original Llama 3.1-8B arm after the Llama
  evaluation proved too slow for this matrix.
- Each completed configuration is evaluated with three seeds: 1234, 2345,
  and 3456. The four eval-agent tasks are `gsm8k_platinum_cot_llama`,
  `mmlu_cot_llama`, `math_500`, and `gpqa_diamond`.
- On the recorded H100/SM90 path, vLLM selected
  `HummingMxFp4LinearKernel` and Humming MoE kernels where applicable. These
  consume the checkpoint's dynamic FP8 activation configuration. The imatrix
  runner checks the checkpoint config and server logs, and rejects an
  evaluation if the activation configuration is missing, a weight-only kernel
  is selected, or vLLM reports that activation quantization was ignored.

### Nemotron status

Nemotron quantization completed, but its MXFP4xFP8 evaluation could not start
successfully, so it is not part of the completed matrices. The Humming MoE
input quantizer requires the expert input width to be divisible by the FP8
group size 128. Nemotron's second expert projection receives width 1856, which
leaves a 64-element tail and triggers Humming's divisibility assertion. The
previous MXFP4A16 run used a different, weight-only kernel path and therefore
does not establish that the xFP8 MoE path works. Attempts to use Marlin and
Triton as xFP8 fallbacks also failed. The runners intentionally use the three
models above rather than counting Nemotron as a successful evaluation.

## Imatrix ablation

The imatrix runner evaluates 48 configurations: six powers (`2.0`, `2.2`,
`2.4`, `2.6`, `2.8`, `3.0`), two objectives, and expanded-observer settings on
all three models, plus the default observer settings on Qwen3.8-27B. Patience
is zero, so the configured candidate list is searched exhaustively.

The objectives are:

- `importance * abs(QDQ(w) - w)^power`
- The same term multiplied by `sqrt(w^2 + group_absmax^2)`
  (`magnitude_normalization=True`)

E8M0 scales make exponential spacing appropriate. The `grid_type` is explicit
in both observer variants so behavior does not depend on automatic defaults:

| Variant | Observer | Grid | Expand | Max shrink | Candidates |
|---|---|---:|---:|---:|---|
| Default | `imatrix_mse` | 100 | 1.0 | 0.95 | `1, 1/2, 1/4, 1/8, 1/16, 1/32, 0.06` |
| Expanded | `nvfp4_expanded_imatrix` | 200 | 1.0 | 63/64 | `1, 1/2, 1/4, 1/8, 1/16, 1/32, 0.025, 1/64` |

The final non-power-of-two candidate in each row preserves the endpoint of the
corresponding linear grid. Qwen3.8-27B uses both variants; Qwen2.5-7B-Instruct
and Qwen3-30B-A3B use the expanded variant.

```bash
IMATRIX=examples/imatrix/run_mxfp4xfp8_imatrix_ablation.py
IMATRIX_RUNS="$PWD/runs/mxfp4xfp8_imatrix_ablation"

"$RHVENV_PYTHON" "$IMATRIX" prepare --root "$IMATRIX_RUNS"
"$RHVENV_PYTHON" "$IMATRIX" smoke --root "$IMATRIX_RUNS" --workers 4
"$RHVENV_PYTHON" "$IMATRIX" full --root "$IMATRIX_RUNS" --workers 6 \
  --seeds 1234,2345,3456
```

The smoke phase omits MMLU because the registry's MMLU smoke workload is very
large. It validates serving and calibrates concurrency before the full run.
The full run is seed-major and retries an evaluation at most once.

## Model-free MSE ablation

The model-free MSE runner evaluates 36 configurations: six powers times
`magnitude_normalization` off/on for each of the three models. It runs
model-free PTQ over the weights using `nvfp4_expanded_mse`; the resulting
checkpoint retains the same MXFP4xFP8 preset, and vLLM evaluation is subject to
the W4A8 config and kernel checks described above.

For each weight, the MSE error is `abs(QDQ(w) - w)^power`. With magnitude
normalization enabled, the observer multiplies that error by
`sqrt(w^2 + group_absmax^2)` before reducing it over the group.

The observer uses norm equal to the selected power, `expand=1.0`,
`maxshrink=63/64`, `grid_type="exponential"`, `grid=8`, and `patience=1000`.
The search checks exactly seven multipliers: `1, 1/2, 1/4, 1/8, 1/16, 1/32,
1/64`. No separate grid-size ablation was run for MSE.

```bash
MSE=examples/imatrix/run_mxfp4xfp8_mse_ablation.py
MSE_RUNS="$PWD/runs/mxfp4xfp8_mse_model_free_ablation"

"$RHVENV_PYTHON" "$MSE" check-config
"$RHVENV_PYTHON" "$MSE" prepare --root "$MSE_RUNS"
"$RHVENV_PYTHON" "$MSE" run --root "$MSE_RUNS" --workers 6
```

The MSE `run` command performs smoke checks, quantizes each configuration,
then completes evaluation seed by seed. It records generated summaries only
inside the selected run directory.
