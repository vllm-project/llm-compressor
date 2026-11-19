# Reproducing the MXFP4xFP8 ablations

This guide documents the imatrix-only, GPTQ+imatrix, and model-free MSE
ablations for MXFP4 weights with dynamic FP8 activations. Results tables are
intentionally not included; each run writes its manifests, logs, status,
checkpoints, and generated summary under its run directory. The MSE runner also
supports MXFP4A16. The separate [MXFP4A16 imatrix guide](MXFP4A16_IMATRIX_REPRODUCTION.md)
documents the already-merged A16 imatrix runner.

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
Run manifests record the pinned model and registry revisions and the applicable
dataset and source revisions.

The `chg` command must be available for GPU reservation. The runners use it
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
  - `google/gemma-4-26B-A4B-it` —
    `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
  - `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` —
    `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`
- Qwen2.5-7B-Instruct replaced the original Llama 3.1-8B arm after the Llama
  evaluation proved too slow for this matrix.
- Each completed configuration is evaluated with three seeds: 1234, 2345,
  and 3456. The four eval-agent tasks are `gsm8k_platinum_cot_llama`,
  `mmlu_cot_llama`, `math_500`, and `gpqa_diamond`.
- The pinned registry configures three repetitions per task, with maximum
  generated-token limits of 8192 for GSM8K Platinum and MMLU, 16384 for
  Math500, and 65536 for GPQA Diamond. These are ceilings; actual generation
  lengths vary by model and prompt. The GPQA setting can make reasoning-model
  evaluations take considerably longer than the shorter instruct tasks.
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
Triton as xFP8 fallbacks also failed. The full-dense xFP8 imatrix and MSE
matrices therefore use the three Qwen models. The GPTQ matrix and optional MSE
scope matrix include Nemotron attention-only; its expert weights remain dense,
so this does not exercise the failing xFP8 expert kernel. The separate
scope-targeted imatrix extension uses a Nemotron dense-only arm, which leaves
the expert weights dense. Nemotron expert-only is excluded.

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
| Default | `imatrix_mse` | 100 | 1.0 | 0.95 | `1, 1/2, 1/4, 1/8, 1/16, 0.06` |
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

### Scope-targeted imatrix extension

The same runner supports the follow-up scope study. Its default scope selection
reproduces the two arms in the current extension matrix: Nemotron 3.5
Lightning-30B-A3B dense-only and Gemma 4 26B-A4B-it expert-only. It uses the
same six powers, two objectives, expanded observer, 512-sample calibration,
and evaluation seeds as the main imatrix matrix. The observer grid uses the
expanded MXFP4xFP8 settings above.

```bash
export MXFP4_IMATRIX_SCOPE_ABLATION=1
SCOPE_RUNS="$PWD/runs/mxfp4xfp8_imatrix_scope_extension_ablation"
"$RHVENV_PYTHON" examples/imatrix/run_mxfp4xfp8_imatrix_ablation.py prepare \
  --root "$SCOPE_RUNS"
"$RHVENV_PYTHON" examples/imatrix/run_mxfp4xfp8_imatrix_ablation.py smoke \
  --root "$SCOPE_RUNS" --workers 6
"$RHVENV_PYTHON" examples/imatrix/run_mxfp4xfp8_imatrix_ablation.py full \
  --root "$SCOPE_RUNS" --workers 6 --seeds 1234,2345,3456
```

Set `MXFP4_ABLATION_INCLUDE_MODELS` before `prepare` to select alternate
scope arms. Supported keys are `qwen3_30b_a3b_expert_only`,
`qwen3_30b_a3b_dense_only`, `nemotron35_30b_a3b_dense_only`,
`gemma4_26b_a4b_expert_only`, and `gemma4_26b_a4b_dense_only`. For example,
Qwen3 expert-only can be run by setting
`MXFP4_ABLATION_INCLUDE_MODELS=qwen3_30b_a3b_expert_only`. Smoke-test a new
scope selection before starting the full seeds; Gemma dense-only was not part
of the recorded default extension matrix.

Nemotron expert-only is intentionally not a runnable evaluation arm. Its
1856-wide expert projection is not divisible by Humming's dynamic FP8 group
size of 128, so the W4A8 MoE kernel fails its input-alignment assertion. The
Nemotron dense-only arm leaves routed and shared experts in BF16 and quantizes
the supported dense Linear projections. Scope selection is recorded in each
run manifest and appears as a separate `scope` column in the generated result
tables.

## GPTQ plus imatrix ablation

The GPTQ runner applies `GPTQModifier` with the same MXFP4xFP8 preset,
calibration set, power values, objectives, and evaluation tasks. It reproduces
72 model/scope/observer configurations: Qwen2.5-7B-Instruct and Qwen3.8-27B
dense, Qwen3-30B-A3B experts-only and attention-only, and Nemotron 3.5
Lightning-30B-A3B attention-only. Qwen3.8-27B uses both observer settings;
the other arms use the expanded observer. Nemotron experts-only is omitted.
Each configuration is evaluated with seeds 1234, 2345, and 3456.

```bash
GPTQ_IMATRIX=examples/imatrix/run_mxfp4xfp8_gptq_imatrix_ablation.py
GPTQ_RUNS="$PWD/runs/mxfp4xfp8_gptq_imatrix_ablation"

"$RHVENV_PYTHON" "$GPTQ_IMATRIX" prepare --root "$GPTQ_RUNS"
"$RHVENV_PYTHON" "$GPTQ_IMATRIX" smoke --root "$GPTQ_RUNS" --workers 6
"$RHVENV_PYTHON" "$GPTQ_IMATRIX" full --root "$GPTQ_RUNS" --workers 6
```

The default observer uses the exponential `1` through `1/16` candidates and
retains the final linear-grid endpoint. The expanded observer searches the
exponential `1` through `1/64` candidates and retains its final endpoint. Both
use the observer names, grid sizes, and max-shrink values listed in the table
above, with `patience=0`, `expand=1`, and `norm=power`. A run manifest records
the exact observer arguments and pinned revisions.

## Model-free MSE ablation

The model-free MSE runner evaluates 36 configurations per scheme: six powers
times `magnitude_normalization` off/on for each of Qwen2.5-7B-Instruct,
Qwen3.8-27B, and Qwen3-30B-A3B. It runs model-free PTQ over the weights using
`nvfp4_expanded_mse`. With MXFP4xFP8, checkpoints retain dynamic FP8
activations and evaluation uses the W4A8 checks above. With MXFP4A16,
checkpoints are weight-only and evaluation verifies the Marlin MXFP4 path.

For each weight, the MSE error is `abs(QDQ(w) - w)^power`. With magnitude
normalization enabled, the observer multiplies that error by
`sqrt(w^2 + group_absmax^2)` before reducing it over the group.

The observer uses norm equal to the selected power, `expand=1.0`,
`maxshrink=63/64`, `grid_type="exponential"`, `grid=8`, and `patience=1000`.
The search checks exactly seven multipliers: `1, 1/2, 1/4, 1/8, 1/16, 1/32,
1/64`. No separate grid-size ablation was run for MSE.

```bash
MSE=examples/imatrix/run_mxfp4_mse_model_free_ablation.py

export MXFP4_ABLATION_SCHEME=MXFP4XFP8
MSE_RUNS="$PWD/runs/mxfp4xfp8_mse_model_free_ablation"
"$RHVENV_PYTHON" "$MSE" check-config
"$RHVENV_PYTHON" "$MSE" prepare --root "$MSE_RUNS"
"$RHVENV_PYTHON" "$MSE" run --root "$MSE_RUNS" --workers 6

export MXFP4_ABLATION_SCHEME=MXFP4A16
MSE_RUNS="$PWD/runs/mxfp4a16_mse_model_free_ablation"
"$RHVENV_PYTHON" "$MSE" check-config
"$RHVENV_PYTHON" "$MSE" prepare --root "$MSE_RUNS"
"$RHVENV_PYTHON" "$MSE" run --root "$MSE_RUNS" --workers 6
```

The MSE `run` command performs smoke checks, quantizes each configuration,
then completes evaluation seed by seed. Set `MXFP4_MSE_RUN_ROOT` to override
the default path for the selected scheme. Generated summaries stay inside the
selected run directory. Checkpoints are stored below `HF_HOME` and exposed in
the run directory through its `checkpoints` symlink. To run the MSE MoE-scope
matrix instead of the three dense models, set `MXFP4_MSE_SCOPE_ABLATION=1` and
use a separate run root; the scope matrix selects Qwen3-30B-A3B experts-only
and attention-only, plus Nemotron attention-only.
