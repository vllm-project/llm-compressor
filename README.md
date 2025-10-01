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

Big updates have landed in LLM Compressor! To get a more in-depth look, check out the [deep-dive](https://x.com/RedHat_AI/status/1937865425687093554).

Some of the exciting new features include:

* **Qwen3 Next and Qwen3 VL MoE Quantization Support**: Quantize the Qwen3 Next and Qwen3 VL MoE models and seamlessly run the models in vLLM. Examples for [NVFP4](examples/quantization_w4a4_fp4/qwen3_next_example.py) and [FP8](examples/quantization_w8a8_fp8/qwen3_next_example.py) Quantization have been added for the Qwen3-Next-80B-A3B-Instruct. For the Qwen3 VL MoE, support has been added for the datafree pathway, specifically [FP8 Quantization](examples/quantization_w8a8_fp8/qwen3_vl_moe_fp8_example.py) (e.g channel-wise and block-wise quantization). NOTE: these models are not supported in tranformers<=4.56.2. You may need to install transformers from source.
* **Quantization with Multiple Modifiers**: Multiple quantization modifiers can now be applied to the same model for mixed-precision quantization, for example applying AWQ W4A16 to a model's `self_attn` layers and GPTQ W8A8 to its `mlp` layers. This is an advanced usage of `llm-compressor` and an active area of research. See the [non-uniform quantization support](examples/quantization_non_uniform) section for more detail and [example usage](examples/quantization_non_uniform/quantization_multiple_modifiers.py).
* **QuIP and SpinQuant-style Transforms**: The newly added [`QuIPModifier`](examples/transform/quip_example.py) and [`SpinQuantModifier`](examples/transform/spinquant_example.py) allow users to quantize their models after injecting hadamard weights into the computation graph, reducing quantization error and greatly improving accuracy recovery for low bit weight and activation quantization.
* **DeepSeekV3-style Block Quantization Support**:  This allows for more efficient compression of large language models without needing a calibration dataset. Quantize a Qwen3 model to [W8A8](examples/quantization_w8a8_fp8/fp8_block_example.py). 
* **Llama4 Quantization Support**: Quantize a Llama4 model to [W4A16](examples/multimodal_vision/llama4_example.py) or [NVFP4](examples/quantization_w4a4_fp4/llama4_example.py). The checkpoint produced can seamlessly run in vLLM.
* **FP4 Quantization - now with MoE and non-uniform support:** Quantize weights and activations to FP4 and seamlessly run the compressed model in vLLM. Model weights and activations are quantized following the NVFP4 [configuration](https://github.com/neuralmagic/compressed-tensors/blob/f5dbfc336b9c9c361b9fe7ae085d5cb0673e56eb/src/compressed_tensors/quantization/quant_scheme.py#L104). See examples of [fp4 activation support](examples/quantization_w4a4_fp4/llama3_example.py), [MoE support](examples/quantization_w4a4_fp4/qwen_30b_a3b.py), and [Non-uniform quantization support](examples/quantization_non_uniform) where some layers are selectively quantized to fp8 for better recovery. You can also mix other quantization schemes, such as int8 and int4.

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
* [Quantizing MoE LLMs](examples/quantizing_moe/README.md)
* [Quantizing Vision-Language Models](examples/multimodal_vision/README.md)
* [Quantizing Audio-Language Models](examples/multimodal_audio/README.md)
* [Quantizing Models Non-uniformly](examples/quantization_non_uniform/README.md)

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
