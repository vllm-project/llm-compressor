# <img width="40" alt="tool icon" src="https://github.com/user-attachments/assets/f9b86465-aefa-4625-a09b-54e158efcf96" />  LLM Compressor
`llmcompressor` is an easy-to-use library for optimizing models for deployment with `vllm`, including:

* Comprehensive set of quantization algorithms for weight-only and activation quantization
* Seamless integration with Hugging Face models and repositories
* `safetensors`-based file format compatible with `vllm`
* Large model support via `accelerate`

**✨ Read the announcement blog [here](https://neuralmagic.com/blog/llm-compressor-is-here-faster-inference-with-vllm/)! ✨**

<p align="center">
   <img alt="LLM Compressor Flow" src="https://github.com/user-attachments/assets/adf07594-6487-48ae-af62-d9555046d51b" width="80%" />
</p>

### Supported Formats
* Activation Quantization: W8A8 (int8 and fp8)
* Mixed Precision: W4A16, W8A16
* 2:4 Semi-structured and Unstructured Sparsity

### Supported Algorithms
* Simple PTQ
* GPTQ
* SmoothQuant
* SparseGPT

### When to Use Which Optimization

#### PTQ
PTQ is performed to reduce the precision of quantizable weights (e.g., linear layers) to a lower bit-width. Supported formats are:

##### [W4A16](./examples/quantization_w4a16/README.md)
- Uses GPTQ to compress weights to 4 bits. Requires calibration dataset.
- Useful speed ups in low QPS regimes with more weight compression. 
- Recommended for any GPUs types. 
##### [W8A8-INT8](./examples/quantization_w8a8_int8/README.md)
- Uses channel-wise quantization to compress weights to 8 bits using GPTQ, and uses dynamic per-token quantization to compress activations to 8 bits. Requires calibration dataset for weight quantization. Activation quantization is carried out during inference on vLLM.
- Useful for speed ups in high QPS regimes or offline serving on vLLM. 
- Recommended for NVIDIA GPUs with compute capability <8.9 (Ampere, Turing, Volta, Pascal, or older). 
##### [W8A8-FP8](./examples/quantization_w8a8_fp8/README.md)
- Uses channel-wise quantization to compress weights to 8 bits, and uses dynamic per-token quantization to compress activations to 8 bits. Does not require calibration dataset. Activation quantization is carried out during inference on vLLM.
- Useful for speed ups in high QPS regimes or offline serving on vLLM. 
- Recommended for NVIDIA GPUs with compute capability >8.9 (Hopper and Ada Lovelace). 

#### Sparsification
Sparsification reduces model complexity by pruning selected weight values to zero while retaining essential weights in a subset of parameters. Supported formats include:

##### [2:4-Sparsity with FP8 Weight, FP8 Input Activation](./examples/sparse_2of4_quantization_fp8/README.md)
- Uses (1) semi-structured sparsity (SparseGPT), where, for every four contiguous weights in a tensor, two are set to zero. (2) Uses channel-wise quantization to compress weights to 8 bits and dynamic per-token quantization to compress activations to 8 bits.
- Useful for better inference than W8A8-fp8, with almost no drop in its evaluation score [blog](https://neuralmagic.com/blog/24-sparse-llama-fp8-sota-performance-for-nvidia-hopper-gpus/). Note: Small models may experience accuracy drops when the remaining non-zero weights are insufficient to recapitulate the original distribution.
- Recommended for compute capability >8.9 (Hopper and Ada Lovelace).


## Installation

```bash
pip install llmcompressor
```

## Get Started

### End-to-End Examples

Applying quantization with `llmcompressor`:
* [Activation quantization to `int8`](examples/quantization_w8a8_int8/README.md)
* [Activation quantization to `fp8`](examples/quantization_w8a8_fp8/README.md)
* [Weight only quantization to `int4`](examples/quantization_w4a16/README.md)
* [Quantizing MoE LLMs](examples/quantizing_moe/README.md)
* [Quantizing Vision-Language Models](examples/multimodal_vision/README.md)
* [Quantizing Audio-Language Models](examples/multimodal_audio/README.md)

### User Guides
Deep dives into advanced usage of `llmcompressor`:
* [Quantizing with large models with the help of `accelerate`](examples/big_models_with_accelerate/README.md)


## Quick Tour
Let's quantize `TinyLlama` with 8 bit weights and activations using the `GPTQ` and `SmoothQuant` algorithms.

Note that the model can be swapped for a local or remote HF-compatible checkpoint and the `recipe` may be changed to target different quantization algorithms or formats.

### Apply Quantization
Quantization is applied by selecting an algorithm and calling the `oneshot` API.

```python
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

# Select quantization algorithm. In this case, we:
#   * apply SmoothQuant to make the activations easier to quantize
#   * quantize the weights to int8 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]

# Apply quantization using the built in open_platypus dataset.
#   * See examples for demos showing how to pass a custom calibration set
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
output = model.generate("My name is")
```

## Questions / Contribution

- If you have any questions or requests open an [issue](https://github.com/vllm-project/llm-compressor/issues) and we will add an example or documentation.
- We appreciate contributions to the code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here](CONTRIBUTING.md).
