# LLM Compressor
`llmcompressor` is an easy-to-use library for optimizing models for deployment with `vllm`, including:

* Comprehensive set of quantization algorithms for weight-only and activation quantization
* Seamless integration with Hugging Face models and repositories
* `safetensors`-based file format compatible with `vllm`
* Large model support via `accelerate`

<p align="center">
   <img alt="LLM Compressor Flow" src="docs/images/architecture.png" width="75%" />
</p>


### Supported Formats
* Activation Quantization: W8A8 (int8 and fp8)
* Mixed Precision: W4A16, W8A16
* 2:4 Semi-structured and unstructured Sparsity

### Supported Algorithms
* PTQ (Post Training Quantization)
* GPTQ
* SmoothQuant
* SparseGPT


## Installation

```bash
pip install llmcompressor
```

## Get Started

### End-to-End Examples

Applying quantization with `llmcompressor`:
* [Activation quantization to `int8`](examples/quantization_w8a8_int8)
* [Activation quantization to `fp8`](examples/quantization_w8a8_fp8)
* [Weight only quantization to `int4`](examples/quantization_w4a16)

### User Guides
Deep dives into advanced usage `llmcompressor`:
* [Quantizing with large models with the help of `accelerate`](examples/big_models_with_accelerate)


## Quick Tour
The following example shows how to quantize `TinyLlama/TinyLlama-1.1B-Chat-v1.0` to 8-bit weights and activations using the `GPTQ` and `SmoothQuant` algorithms. Note that the model can be swapped for a local or remote HF-compatible checkpoint and the `recipe` may be changed to target different quantization algorithms or formats.

### Apply Quantization
Quantization is applied by selecting an algorithm and calling the `oneshot` API.

```python
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot

# Select quantization algorithm. In this case, we:
#   * apply SmoothQuant to make the activations easier to quantize
#   * quantize the weights to int8 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]

# Apply quantization using the open_platypus dataset.
# See examples for demos showing how to pass a custom calibration set.
oneshot(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="TinyLlama-1.1B-Chat-v1.0-INT8",
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

### Inference with vLLM

The checkpoints created by `llmcompressor` can be loaded and run in `vllm`:

Install:

```bash
pip install vllm
```

Run:
```python
from vllm import LLM

model = LLM("TinyLlama-1.1B-Chat-v1.0-INT8")
output = model.generate("I love quantization in vllm because")
```

## Questions / Contribution

- If you have any questions or requests open an [issue](https://github.com/vllm-project/llm-compressor/issues) and we will add an example or documentation.
- We appreciate contributions to the code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here](CONTRIBUTING.md).
