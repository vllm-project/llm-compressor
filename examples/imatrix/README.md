# iMatrix Importance-Weighted Quantization

`imatrix_mse` is an observer that uses per-channel activation importance (E[x²]) to weight quantization error during range selection. Channels that carry more signal get more careful range optimization.

Two components work together:
- **`IMatrixGatherer`**: triggers a calibration pass so the observer can collect importance data
- **`imatrix_mse` observer**: collects E[x²] per input channel via forward pre-hooks and uses importance weighting in the MSE grid search: `err = sum(importance * |Q(w) - w|^p)`

> See [RFC #2456](https://github.com/vllm-project/llm-compressor/discussions/2456) for the full design discussion.

## Quickstart

```bash
python3 llama3_imatrix_example.py
```

The simplest setup uses `preset_name_to_scheme` to configure W4A16 and swaps in the `imatrix_mse` observer:

```python
from compressed_tensors.quantization import preset_name_to_scheme
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer

scheme = preset_name_to_scheme("W4A16", ["Linear"])
scheme.weights.observer = "imatrix_mse"

recipe = [
    IMatrixGatherer(ignore=["lm_head"]),
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
    IMatrixGatherer(ignore=["lm_head"]),
    GPTQModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    ),
]
```

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
