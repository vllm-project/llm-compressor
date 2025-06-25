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

## 🚀 What's New!

Big updates have landed in LLM Compressor! To get a more in-depth look, check out the [deep-dive](https://x.com/RedHat_AI/status/1937865425687093554).

Some of the exciting new features include:

* **Large Model Support with Sequential Onloading** As of llm-compressor>=0.6.0, you can now quantize very large language models on a single GPU. Models are broken into disjoint layers which are then onloaded to the GPU one layer at a time. For more information on sequential onloading, see [Big Modeling with Sequential Onloading](examples/big_models_with_sequential_onloading/README.md) as well as the [DeepSeek-R1 Example](examples/quantizing_moe/deepseek_r1_example.py).
* **Preliminary FP4 Quantization Support:** Quantize weights and activations to FP4 and seamlessly run the compressed model in vLLM. Model weights and activations are quantized following the NVFP4 [configuration](https://github.com/neuralmagic/compressed-tensors/blob/f5dbfc336b9c9c361b9fe7ae085d5cb0673e56eb/src/compressed_tensors/quantization/quant_scheme.py#L104). See examples of [weight-only quantization](examples/quantization_w4a16_fp4/llama3_example.py) and [fp4 activation support](examples/quantization_w4a4_fp4/llama3_example.py). Support is currently preliminary and additional support will be added for MoEs.
* **Updated AWQ Support:** Improved support for MoEs with better handling of larger models
* **Axolotl Sparse Finetuning Integration:** Seamlessly finetune sparse LLMs with our Axolotl integration. Learn how to create [fast sparse open-source models with Axolotl and LLM Compressor](https://developers.redhat.com/articles/2025/06/17/axolotl-meets-llm-compressor-fast-sparse-open). See also the [Axolotl integration docs](https://docs.axolotl.ai/docs/custom_integrations.html#llmcompressor).
* **Day 0 Llama 4 Support:** Meta utilized LLM Compressor to create the [FP8-quantized Llama-4-Maverick-17B-128E](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8), optimized for vLLM inference using [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) format.

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

### When to Use Which Optimization

Please refer to [docs/schemes.md](./docs/schemes.md) for detailed information about available optimization schemes and their use cases.


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
