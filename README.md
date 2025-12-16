<div align="center">

<h1>
  <img width="40" alt="tool icon" src="https://github.com/user-attachments/assets/f9b86465-aefa-4625-a09b-54e158efcf96" />
  <span style="font-size:80px;">LLM Compressor</span>
</h1>

[![docs](https://img.shields.io/badge/docs-LLM--Compressor-blue)](https://docs.vllm.ai/projects/llm-compressor/en/latest/) [![PyPI](https://img.shields.io/pypi/v/llmcompressor.svg)](https://pypi.org/project/llmcompressor/)

</div>

`llmcompressor` is an easy-to-use library for optimizing models for deployment with `vllm`, including:

* Comprehensive set of quantization algorithms for weight-only and activation quantization
* Seamless integration with Hugging Face models and repositories
* `safetensors`-based file format compatible with `vllm`
* Large model support via `accelerate`

**✨ Read the announcement blog [here](https://neuralmagic.com/blog/llm-compressor-is-here-faster-inference-with-vllm/)! ✨**

<p align="center">
   <img alt="LLM Compressor Flow" src="https://github.com/user-attachments/assets/adf07594-6487-48ae-af62-d9555046d51b" width="80%" />
</p>

---

💬 Join us on the [vLLM Community Slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack) and share your questions, thoughts, or ideas in:

- `#sig-quantization`
- `#llm-compressor`

---

## 🚀 What's New!

Big updates have landed in LLM Compressor! To get a more in-depth look, check out the [LLM Compressor overview](https://docs.google.com/presentation/d/1WNkYBKv_CsrYs69lb7bJKjh2dWt8U1HXUw7Gr4Wn3gE/edit?usp=sharing).

Some of the exciting new features include:

* **Batched Calibration Support**: LLM Compressor now supports calibration with batch sizes > 1. A new [`batch_size`](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/args/dataset_arguments.py#L70) argument has been added to the `dataset_arguments` enabling the option to improve quantization speed. Default `batch_size` is currently set to 1
* **New Model-Free PTQ Pathway**: A new model-free PTQ pathway has been added to LLM Compressor, called [`model_free_ptq`](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/entrypoints/model_free/__init__.py#L36). This pathway allows you to quantize your model without the requirement of Hugging Face model definition and is especially useful in cases where `oneshot` may fail. This pathway is currently supported for data-free pathways only i.e FP8 quantization and was leveraged to quantize the [Mistral Large 3 model](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512). Additional [examples](https://github.com/vllm-project/llm-compressor/tree/main/examples/model_free_ptq) have been added illustrating how LLM Compressor can be used for Kimi K2
* **Extended KV Cache and Attention Quantization Support**: LLM Compressor now supports attention quantization. KV Cache quantization, which previously only supported per-tensor scales, has been extended to support any quantization scheme including a new `per-head` quantization scheme. Support for these checkpoints is on-going in vLLM and scripts to get started have been added to the [experimental folder](https://github.com/vllm-project/llm-compressor/blob/main/experimental/llama3_attention.py)
* **Generalized AWQ Support**: The AWQModifier has been updated to support quantization schemes beyond W4A16 (e.g W4AFp8). In particular, AWQ no longer constrains that the quantization config needs to have the same settings for `group_size`, `symmetric`, and `num_bits` for each config_group
* **AutoRound Quantization Support**: Added [`AutoRoundModifier`](examples/autoround/llama3_example.py) for quantization using [AutoRound](https://aclanthology.org/2024.findings-emnlp.662.pdf), an advanced post-training algorithm that optimizes rounding and clipping ranges through sign-gradient descent. This approach combines the efficiency of post-training quantization with the adaptability of parameter tuning, delivering robust compression for large language models while maintaining strong performance
* **Experimental MXFP4 Support**: Models can now be quantized using an [`MXFP4`](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py#L208) pre-set scheme. Examples can be found under the [experimental folder](https://github.com/vllm-project/llm-compressor/blob/main/experimental/mxfp4/llama3_mxfp4.py). This pathway is still experimental as support and validation with vLLM is still a WIP. 
* **R3 Transform Support**: LLM Compressor now supports applying transforms to attention in the style of SpinQuant's R3 rotation. Note: this feature is currently not yet supported in vLLM

### Supported Formats
* Activation Quantization: W8A8 (int8 and fp8)
* Mixed Precision: W4A16, W8A16, NVFP4 (W4A4 and W4A16 support)
* 2:4 Semi-structured and Unstructured Sparsity

### Supported Algorithms
* Simple PTQ
* GPTQ
* AWQ
* SmoothQuant
* SparseGPT
* AutoRound

### When to Use Which Optimization

Please refer to [compression_schemes.md](./docs/guides/compression_schemes.md) for detailed information about available optimization schemes and their use cases.


## Installation

```bash
pip install llmcompressor
```

## Get Started

### End-to-End Examples

Applying quantization with `llmcompressor`:
* [Activation quantization to `int8`](examples/quantization_w8a8_int8/README.md)
* [Activation quantization to `fp8`](examples/quantization_w8a8_fp8/README.md)
* [Activation quantization to `fp4`](examples/quantization_w4a4_fp4/llama3_example.py)
* [Weight only quantization to `fp4`](examples/quantization_w4a16_fp4/llama3_example.py)
* [Weight only quantization to `int4` using GPTQ](examples/quantization_w4a16/README.md)
* [Weight only quantization to `int4` using AWQ](examples/awq/README.md)
* [Weight only quantization to `int4` using AutoRound](examples/autoround/README.md)
* [Quantizing MoE LLMs](examples/quantizing_moe/README.md)
* [Quantizing Vision-Language Models](examples/multimodal_vision/README.md)
* [Quantizing Audio-Language Models](examples/multimodal_audio/README.md)
* [Quantizing Models Non-uniformly](examples/quantization_non_uniform/README.md)

### User Guides
Deep dives into advanced usage of `llmcompressor`:
* [Quantizing large models with sequential onloading](examples/big_models_with_sequential_onloading/README.md)


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

## Citation

If you find LLM Compressor useful in your research or projects, please consider citing it:

```bibtex
@software{llmcompressor2024,
    title={{LLM Compressor}},
    author={Red Hat AI and vLLM Project},
    year={2024},
    month={8},
    url={https://github.com/vllm-project/llm-compressor},
}
```
