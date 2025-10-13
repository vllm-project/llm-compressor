# Applying Transforms to Improve Quantization Accuracy

This directory contains example scripts for applying transforms to models for the purpose of improving quantization accuracy. For more information on transforms, see [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456). The two transform styles currently supported are SpinQuant/QuaRot-style (`SpinQuantModifier`), and QuIP-style (`QuIPModifier`).

See also [[vLLM Office Hours #31] vLLM and LLM Compressor Update - August 28, 2025](https://www.youtube.com/watch?v=WVenRmF4dPY&list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3&index=3).

## Installation

To get started, install the necessary dependencies by executing the following commands:

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## Quickstart

The provided example script demonstrates the process for applying quip-style transforms before quantization.

```bash
python3 quip_example.py
```

### Step 1: Select a Model, Dataset, and Recipe

In this step, you'll choose a base model for quantization and a transformation + quantization recipe.

- **Models**: Can be referenced from a local directory or retrieved from the Hugging Face Hub.
- **Recipes**: These are YAML files or Python modifier objects that describe how a model should be optimized during or after training. In this example, we use the `QuIPModifier` applied before the `QuantizationModifier` with the scheme set to `FP8`.

```python
from llmcompressor.modifiers.transform import QuIPModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = [
    QuIPModifier(
        rotations=["v", "u"], transform_block_size=128, transform_type="hadamard"
    ),
    QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]
```

Note that `QuIPModifier` can be customized. For a full list of the available arguments, see the [docstring](/src/llmcompressor/modifiers/transform/spinquant/base.py) or documentation.

* `rotations` determines which of the input rotation (v) or output rotations (u) should be used.
* `transform_block_size` determines the size of the hadamard. Smaller hadamards require less cost at runtime.
* `transform_type` determines how the transform is constrcted. hadamard uses the sylvester construction.

### Step 2: Run Quantization Using Oneshot

The `oneshot` method applies the selected recipe to your model and dataset without requiring any fine-tuning. The model will be quantized and saved to `Llama-3.1-8B-Instruct-quip-w4a16`. We use the "datafree" pipeline, since our recipe does not require calibration data.

```python
from llmcompressor import oneshot

# Apply algorithms.
oneshot(model=model, recipe=recipe, pipeline="datafree")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-quip-w4a16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

### Step 3: Run optimized model in vLLM
Models optimized with the `hadamard` transform type will be able to leverage the hadacore kernels for accelerated inference. Use the [benchmarks/latency.py](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/latency.py) script to benchmark latency

```bash
python3 benchmarks/benchmark_latency.py --model path/to/Llama-3.2-1B-Instruct-quip-w4a16
```


#### Dense Model Latency (sec) ####
| [Base](https://huggingface.co/meta-llama/Llama-3.2-1B-instruct) | Hadacore | GEMM |
| - | - | - |
| 0.4710 | 0.4948 | 1.3946 |

#### Quantized Model Latency (sec) ####
| Base W4A16 | Hadacore | GEMM |
| - | - | - |
| 0.4402 | 0.4489 | 1.2917 |