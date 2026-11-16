# iMatrix Importance-Weighted Quantization

`imatrix_mse` is an observer that uses per-channel activation importance (E[x²]) to weight quantization error during range selection. Channels that carry more signal get more careful range optimization.

When configured as a weight observer, `imatrix_mse` collects E[x²] per input channel during calibration via forward pre-hooks and uses importance weighting in the MSE grid search: `err = sum(importance * |Q(w) - w|^p)`.

> See [RFC #2456](https://github.com/vllm-project/llm-compressor/issues/2456) for the full design discussion.

## Quickstart

```bash
python3 llama3_imatrix_example.py
```

The simplest setup uses `preset_name_to_scheme` to configure W4A16 and swaps in the `imatrix_mse` observer:

```python
from compressed_tensors.quantization import preset_name_to_scheme
from llmcompressor.modifiers.quantization import QuantizationModifier

scheme = preset_name_to_scheme("W4A16", ["Linear"])
scheme.weights.observer = "imatrix_mse"

recipe = [
    QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    ),
]
```

## Composing with GPTQ

iMatrix composes with GPTQ by providing importance-weighted ranges for the Hessian-based rounding:

```python
from llmcompressor.modifiers.gptq import GPTQModifier

scheme = preset_name_to_scheme("W4A16", ["Linear"])
scheme.weights.observer = "imatrix_mse"

recipe = [
    GPTQModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    ),
]
```

## Groupmax objectives

The `imatrix_mse` observer supports three objectives. `basic` uses
`importance * abs(Q(w) - w) ** norm`; `groupmax` multiplies that weight by
`sqrt(w ** 2 + groupmax ** 2)`, and `groupmax_div` divides by the same term.
`groupmax` is the original absolute maximum in the weight's quantization group.
The groupmax variants require grouped weight quantization, as in NVFP4.

```python
scheme = preset_name_to_scheme("NVFP4", ["Linear"])
scheme.weights.observer = "imatrix_mse"
scheme.weights.observer_kwargs = {"norm": 2.4, "objective": "groupmax_div"}
```

## Reproducing NVFP4 iMatrix quantization and evaluation

The scripts below quantize one model/power/objective/profile at a time. They
record the model, calibration recipe, observer settings, selected modules, and
library versions in `imatrix_ablation_config.json` beside each checkpoint.
Model profiles include the NVFP4 exclusions used for these experiments; in
particular, routers and `lm_head` are validated to remain unquantized.

For a one-shot imatrix run:

```bash
python examples/imatrix/quantize_nvfp4_imatrix_variant.py \
  --model qwen3_30b_a3b \
  --power 2.4 \
  --objective groupmax_div \
  --module-profile all \
  --variant-dir ./runs/qwen3-30b-p2.4-groupmax-div
```

For GPTQ plus the same observer:

```bash
python examples/imatrix/quantize_nvfp4_gptq_imatrix_variant.py \
  --model qwen3_30b_a3b \
  --power 2.4 \
  --objective groupmax_div \
  --module-profile all \
  --variant-dir ./runs/qwen3-30b-p2.4-groupmax-div-gptq
```

`--model` accepts `qwen38_27b`, `qwen3_30b_a3b`,
`nemotron35_lightning_30b_a3b`, `gemma4_26b_a4b_it`, and
`qwen25_7b_instruct`. `--module-profile attention_only` is supported for
Qwen3-30B-A3B, Nemotron 3.5 Lightning, and Gemma 4. The shipped recipes use
the `perfectblend` `train[:512]` calibration split with a 4096-token sequence
limit and calibrate all MoE experts where supported. Nemotron uses NVFP4A16
with unquantized inputs, following its model-specific recipe.

For the benchmark suite, use the `nvfp4-imatrix-ablation` skill in
`.agents/skills` together with the external `eval-agent` skill. The wrapper
`evaluate_nvfp4_imatrix_checkpoint.py` runs the unchanged eval-agent registry
tasks: strict GSM8K, MMLU, Math-500 pass@1, GPQA Diamond pass@1, and IFEval.
It runs MMLU only for seed 1234 and the other tasks for seeds 1234, 2345, and
3456. The wrapper accepts a complete `--server-cmd`, so vLLM flags, chat
templates, and GPU reservation remain explicit and reproducible. For Gemma 4
attention-only checkpoints it also repairs the runtime quantization targets
for vLLM's HF path mapping and shared K/V fusion before serving.

## Results

W4A16, Llama-3.1-8B, group_size=128, WikiText-2 token-level perplexity (141 chunks x 2048):

| Config | PPL |
|---|---|
| FP16 baseline | 6.24 |
| RTN `memoryless_minmax` | 6.96 |
| GPTQ | 6.92 |
| AWQ | 6.89 |
| RTN `imatrix_mse` | 6.85 |
| GPTQ + `imatrix_mse` | 6.83 |

## Questions or Feature Request?

Please open up an issue on `vllm-project/llm-compressor`
