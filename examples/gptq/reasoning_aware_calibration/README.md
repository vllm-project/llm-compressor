# Reasoning-aware calibration for GPTQ

Reasoning models spend most of their inference time generating their own
chain-of-thought, but calibration sets usually contain responses written by
someone else. [Reasoning-Aware Compression (RAC)](https://arxiv.org/abs/2509.12464)
proposes calibrating on the model's *own* reasoning traces instead: generate
responses with the uncompressed model, then calibrate on `prompt + generated
response` with the statistics restricted to the generated tokens.

llm-compressor needs no changes for this. Trace generation is done with vLLM,
and GPTQ can exclude tokens from its Hessian through
`oneshot(..., use_loss_mask=True)` and a `loss_mask` column in the dataset.
This example wires the two together and compares three calibration conditions
on Qwen3-8B.

## Calibration conditions

All conditions use the same 500
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) problems and
exactly the same number of active Hessian tokens.

| Condition | Calibration text | Tokens in the GPTQ Hessian |
|---|---|---|
| `full` | prompt + dataset solution | all tokens (regular GPTQ) |
| `dataset_masked` | prompt + dataset solution | solution tokens only |
| `onpolicy_masked` | prompt + the BF16 model's own trace | generated tokens only (RAC) |

Generated traces are much longer than dataset solutions, so without a token
budget a gain from `onpolicy_masked` could come from more calibration tokens
rather than from the traces being on-policy. For every problem, the masked
conditions therefore keep the shorter of the dataset solution and the generated
trace. The `full` condition keeps every prompt and spends the remaining budget
on solution prefixes, proportionally to the same per-problem lengths. Masked
prompt tokens are still forwarded as context.

## Files

| File | What it does | Environment |
|---|---|---|
| `generate_on_policy_traces.py` | loads the pinned MATH-500 revision, stores prompt and dataset-solution token IDs, and generates BF16 traces with vLLM (token IDs are kept as generated, without re-tokenization) | vLLM |
| `calibrate_gptq.py` | builds one token-matched condition and writes a compressed GPTQ checkpoint plus `calibration_stats.json` | llm-compressor |
| `run_experiment.sh` | generates traces if missing, then calibrates the requested scheme/condition pairs, one GPU per job | both |

vLLM and llm-compressor often pin incompatible versions of torch and
transformers, so `run_experiment.sh` takes a separate interpreter per stage:

```bash
GEN_PYTHON=/path/to/vllm-env/bin/python \
QUANT_PYTHON=/path/to/llmcompressor-env/bin/python \
MODEL_PATH=Qwen/Qwen3-8B \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUT=/data/rac-gptq-qwen3-8b \
bash examples/gptq/reasoning_aware_calibration/run_experiment.sh
```

Traces are generated in thinking mode with temperature `0.6`, top-p `0.95`,
top-k `20`, seed `42` and at most `MAX_NEW_TOKENS=8192` new tokens. By default
the script calibrates the six reported conditions, `FP8_DYNAMIC` and
`MXFP4XFP8` (MXFP4 weights with dynamic FP8 activations) times the three
calibration conditions. Select jobs with e.g.
`CONDITIONS="FP8_DYNAMIC/full W4A16/onpolicy_masked"`. Other knobs: `GEN_TP`
(vLLM tensor parallel size for generation) and `MAX_PARALLEL` (concurrent
calibration jobs, at most one per GPU). Existing traces and checkpoints are
reused.

GPTQ runs with the sequential pipeline and offloads cached layer inputs to CPU,
so calibration of an 8B model fits on one GPU, but the long traces need a
corresponding amount of CPU memory.

## Results

Qwen3-8B, evaluated on the 4,500 problems of the
[Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals) MATH task
that are not among the 500 calibration problems, in thinking mode with the
generation settings above. Full completions are scored with Math-Verify;
the paired difference to BF16 is over the same 4,500 problems. Runs by
@HDCharles on an H100 80 GB; the evaluation scripts and the full report are in
[#3548](https://github.com/vllm-project/llm-compressor/pull/3548).

| Checkpoint | Math-Verify accuracy | Paired Δ vs BF16 (95% CI) |
|---|---|---|
| BF16 | 79.29% ± 0.60 pp | baseline |
| FP8_DYNAMIC, `full` | 79.67% ± 0.60 pp | +0.38 pp [−0.65, +1.40] |
| FP8_DYNAMIC, `dataset_masked` | 80.11% ± 0.60 pp | +0.82 pp [−0.20, +1.85] |
| FP8_DYNAMIC, `onpolicy_masked` | 80.51% ± 0.59 pp | +1.22 pp [+0.22, +2.23] |
| MXFP4XFP8, `full` | 79.04% ± 0.61 pp | −0.24 pp [−1.31, +0.82] |
| MXFP4XFP8, `dataset_masked` | 76.82% ± 0.63 pp | −2.47 pp [−3.51, −1.42] |
| MXFP4XFP8, `onpolicy_masked` | 79.33% ± 0.60 pp | +0.04 pp [−1.01, +1.10] |

- With MXFP4 weights, masking the prompt and calibrating on dataset solutions
  loses 2.5 points, while calibrating on the model's own traces at the same
  token budget stays at BF16 accuracy.
- With FP8, all three conditions are within about one point of BF16. The
  FP8 checkpoints scoring slightly above BF16 should be read as no measurable
  loss, not as an improvement.
- Each condition comes from a single trace-generation and evaluation run, so
  the intervals do not cover the variance between generation runs.
- Serving MXFP4XFP8 checkpoints needs a recent vLLM and compressed-tensors;
  the reported runs used vLLM `0.34.0rc1.dev948+gd2da27eb2` and
  compressed-tensors `0.20.1`.

An earlier AWQ version of this experiment (Qwen3-1.7B, W4A16, 150
OpenR1-Math prompts) did not separate the conditions: at an equal token budget,
on-policy traces were not better than the dataset's DeepSeek-R1 responses
(paired McNemar p=0.12). AWQ fits only one scale per input channel, while GPTQ
compensates rounding errors through a Hessian built directly from the kept
tokens, which may explain why the calibration data matters more for GPTQ.
