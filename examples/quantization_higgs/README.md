# HIGGS mixed-precision quantization examples

HIGGS (Heuristic ILP-Guided Grouped Scheme) chooses a quantization scheme for
each matched weight tensor by minimizing a weighted reconstruction-error
objective under optional bitwidth constraints.

All examples follow two steps:

1. `get_higgs_config` reads the model's safetensors, measures each candidate's
   weight MSE, and solves the allocation ILP without instantiating the model.
2. The generated compressed-tensors config is applied with `model_free_ptq` or
   `oneshot`.

## Model-free example

Use the model-free path when all candidates have data-free activation schemes:

```bash
python examples/quantization_higgs/higgs_nvfp4a16_fp8_model_free.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --target-bits 6.0
```

This example compares `NVFP4A16` with `FP8_DYNAMIC`, applies the returned
arbitrary mixed-precision config through `model_free_ptq`, and saves under
`~/hf_hub`. Use `--max-workers` to control converter concurrency.

`llama3_higgs_example.py` is a fixed, minimal version of the same workflow using
`W4A16` and `W8A16`.

## Calibrated examples

Static activation schemes need calibration and therefore use `oneshot`. The
examples load 256 shuffled UltraChat samples with a maximum sequence length of
2,048.

| Example | Candidates | Application |
| --- | --- | --- |
| `higgs_nvfp4_fp8_qmod.py` | `NVFP4`, `FP8_DYNAMIC` | `QuantizationModifier` |
| `higgs_nvfp4_fp8_gptq.py` | `NVFP4`, `FP8_DYNAMIC` | `GPTQModifier` |
| `higgs_nvfp4a16_fp8_qmod.py` | `NVFP4A16`, `FP8_DYNAMIC` | `QuantizationModifier` |
| `higgs_nvfp4a16_fp8_gptq.py` | `NVFP4A16`, `FP8_DYNAMIC` | `GPTQModifier` |

For example:

```bash
python examples/quantization_higgs/higgs_nvfp4_fp8_qmod.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --target-bits 6.0
```

The GPTQ variants use the same HIGGS allocation but optimize weights with
calibration Hessians when applying it.

## Config API

Use `get_higgs_config` directly to change candidates, targets, ignored modules,
or budgets:

```python
from llmcompressor.entrypoints.higgs import get_higgs_config

config = get_higgs_config(
    model_stub="meta-llama/Meta-Llama-3.1-70B-Instruct",
    candidate_schemes=["NVFP4A16", "FP8_DYNAMIC"],
    targets="Linear",
    ignore=["lm_head", "re:.*embed_tokens"],
    target_avg_bitwidth=6.0,
    target_avg_act_bitwidth=None,
    enforce_fused_layer_constraints=True,
    allow_unquantized=True,
)
```

The weight budget is a parameter-weighted upper bound over matched, non-ignored
tensors. Ignored weights do not contribute to the reported target. The optional
activation budget uses the same parameter weighting as a proxy for activation
cost.

`allow_unquantized=True` is the default. It lets the ILP assign a zero-MSE,
16-bit choice. Those tensors remain in their original dtype and do not appear in
the returned quantization config. Set it to `False` when every matched tensor
must be quantized.

Candidate names are compressed-tensors presets. Common examples are:

- `W4A16` and `W8A16`: integer weights with unquantized activations.
- `NVFP4A16`: NVFP4 weights with unquantized activations.
- `NVFP4`: NVFP4 weights and statically calibrated NVFP4 activations.
- `FP8`: static per-tensor FP8 weights and activations.
- `FP8_DYNAMIC`: per-channel FP8 weights and dynamic per-token FP8 activations.

Scheme details and availability come from the installed compressed-tensors
version. A config containing only data-free schemes can be passed to
`model_free_ptq`. Use `oneshot` whenever a selected scheme has static activation
observers.

## Weight and activation budget sweep

`wnam_activation_sweep.py` generates configs across combinations of WNaM
candidates and weight/activation budgets:

```bash
python examples/quantization_higgs/wnam_activation_sweep.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --short Llama-3-8B \
    --schemes W4A8 W4A16 W8A16 \
    --weight-bits 5.0 6.0 \
    --act-bits 10.0 12.0 \
    --config-only
```

The script currently generates and saves allocation JSON files; it does not
apply the configs to model weights.

## Evaluation

Measure WikiText-2 perplexity directly with vLLM prompt log probabilities:

```bash
python examples/quantization_higgs/measure_ppl.py \
    --model ~/hf_hub/Meta-Llama-3.1-8B-Instruct-HIGGS-FP8_DYNAMIC+NVFP4A16-W6.0avg-convert \
    --max-model-len 4096 \
    --tp 1
```

`measure_ppl.py` writes `eval_results.json` into a local model directory. Its
`byte_perplexity` field is token-level perplexity despite the legacy field name.

For a broader lm-evaluation-harness run covering WikiText, GSM8K, and MMLU:

```bash
bash examples/quantization_higgs/eval_higgs.sh MODEL_PATH 1 4096
```

The shell script expects vLLM and `lm_eval` under `~/vllm`.

## Output and runtime

Quantized checkpoints are saved in Hugging Face format with sharded
safetensors and a compressed-tensors `quantization_config` in `config.json`.
Load them with a runtime that supports every selected scheme, such as a
compatible vLLM build.

Runtime depends heavily on model size, candidate count, storage speed, device,
calibration length, and whether GPTQ is used. HIGGS processes checkpoint shards
sequentially for MSE collection, while model-free PTQ can process shards
concurrently with `max_workers`.
