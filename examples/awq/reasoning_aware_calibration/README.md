# AWQ calibrated on on-policy reasoning traces

Reasoning models spend most of their inference time generating chain-of-thought,
but calibration sets usually contain prompts or responses written by someone
else. [Reasoning-Aware Compression (RAC)](https://arxiv.org/abs/2509.12464)
proposes calibrating on the model's *own* reasoning traces instead: generate
responses with the uncompressed model, then calibrate on `prompt + generated
response` with the loss restricted to the generated tokens.

llm-compressor needs no changes for this. Trace generation is done with vLLM,
and AWQ already supports restricting its loss to masked tokens
(`oneshot(..., use_loss_mask=True)`). This example wires the two together
and compares the result against regular AWQ.

## Files

| File | What it does | Environment |
|------|--------------|-------------|
| `generate_on_policy_traces.py` | samples prompts, generates the model's own responses with vLLM and stores `input_ids` + `loss_mask` | vllm |
| `calibrate_awq.py` | builds one calibration set (`default` / `offpolicy` / `onpolicy`, optionally at a fixed loss-token budget) and runs AWQ | llmcompressor |
| `run_experiment.sh` | runs generation, calibration of every set and lm-eval, skipping stages that are already done | both |

vLLM and llm-compressor often pin incompatible versions of torch/transformers,
so `run_experiment.sh` takes a separate interpreter per stage:

```bash
GEN_PYTHON=/path/to/vllm-env/bin/python \
QUANT_PYTHON=/path/to/llmcompressor-env/bin/python \
LM_EVAL=/path/to/vllm-env/bin/lm_eval \
LIMIT=20 bash run_experiment.sh
```

Other knobs: `MODEL`, `DATASET`, `NUM_SAMPLES`, `MAX_NEW_TOKENS`,
`MAX_SEQ_LENGTH`, `SCHEME` (any preset scheme, e.g. `W3A16`), `ARMS`, `TASKS`,
`MAX_GEN_TOKS`, `LIMIT`, `EVAL_ARGS`, and `TP` (vLLM tensor parallel size, for
models that do not fit on one GPU). The defaults reproduce the MMLU-CoT
table below; the MATH-500 experiment uses

```bash
MODEL=Qwen/Qwen3-1.7B DATASET=open-r1/OpenR1-Math-220k NUM_SAMPLES=150 \
MAX_NEW_TOKENS=8192 MAX_SEQ_LENGTH=8704 \
ARMS="onpolicy:500000 offpolicy:500000 default:500000" SCHEME=W4A16 \
TASKS=minerva_math500 EVAL_ARGS="--num_fewshot 0" MAX_GEN_TOKS=8192 \
bash run_experiment.sh
```

`minerva_math500` needs `pip install "math-verify[antlr4_11_0]"` in the
lm-eval environment; read the `math_verify` metric, `exact_match` stays near 0
because of answer formatting.

## Calibration sets

Every set is built from the same prompts, drawn from the calibration dataset.

- **default**: prompt + the dataset's response, no mask. This is regular AWQ.
- **offpolicy**: prompt + the dataset's response, loss on the response only.
- **onpolicy**: prompt + the model's own response (thinking mode, T=0.6),
  loss on the generated tokens only. This is RAC.

Thinking traces are much longer than dataset responses, so a gain of
`onpolicy` over `offpolicy` could come from more calibration tokens rather than
from the traces being on-policy. An arm can therefore be given a loss-token
budget, `<arm>:<N>` or `<arm>:<other arm>`: whole samples are taken in order
until the budget is reached, and `offpolicy` / `default` draw additional dataset
pairs if their prompts run out.

## Results

All results: Qwen3-1.7B, single RTX 4080 SUPER, evaluation in thinking mode
with vLLM. Paired p-values are exact McNemar tests on per-question results.

### MATH-500

Calibration on 150 prompts from
[OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k),
traces of at most 8192 new tokens, every set cut to the same ~510k loss tokens
(onpolicy: 76 traces; offpolicy and default: the same 95 pairs, differing only
in the mask). Evaluated on `minerva_math500`, all 500 problems, 0-shot,
`math_verify`, up to 8192 generated tokens. Standard error is about 0.019.

| Calibration set | W4A16 | W3A16 |
|-----------------|-------|-------|
| BF16 (reference) | 0.812 | 0.812 |
| default (no mask) | 0.736 | 0.048 |
| offpolicy | 0.778 | 0.084 |
| onpolicy | 0.752 | 0.054 |

At W4A16, masking the prompt helps (offpolicy vs default, p=0.009), but
on-policy traces do not improve on dataset responses at an equal token budget
(onpolicy vs offpolicy, p=0.12, in favour of offpolicy). W3A16 breaks the model
regardless of calibration set, so it cannot separate them.

### MMLU-CoT

256 prompts from
[open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend),
traces of at most 2048 new tokens. Evaluated on `mmlu_cot_llama`, limited to
the first 20 questions per subject (1140 questions), strict-match, up to 4096
generated tokens. Standard error is about 0.014.

| Calibration set | Loss tokens | W4A16 |
|-----------------|-------------|-------|
| BF16 (reference) | - | 0.596 |
| default (no mask) | - | 0.559 |
| offpolicy | 75k | 0.570 |
| onpolicy:offpolicy (48 traces) | 75k | 0.561 |
| offpolicy:onpolicy (1461 pairs) | 408k | 0.604 |
| onpolicy | 408k | 0.610 |

At an equal number of loss tokens, on-policy traces and dataset responses are
within noise of each other (408k: p≈0.67). Most of the gain of `onpolicy` over
regular AWQ comes from the larger number of calibration tokens that long
reasoning traces provide.

## Caveats and next steps

- Only a 1.7B model was tested. Larger models (e.g. Qwen3-8B / 32B) and more
  seeds would show whether on-policy traces matter at scale.
- The RAC paper evaluates pruning (SparseGPT at 20-50% sparsity), not
  quantization. W4A16 leaves a gap of only 3-8 points, and W3A16 collapses, so
  an intermediate setting (e.g. W4A16 with a larger group size, or W3A16 with
  more calibration data) may separate the sets better.
- With vLLM, `lm_eval` occasionally keeps running after it has written its
  results; it is safe to stop it once `results_*.json` exists.
