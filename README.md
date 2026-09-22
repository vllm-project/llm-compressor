<div align="center">

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="docs/assets/llmcompressor-icon-name-dark.png"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="docs/assets/llmcompressor-icon-name-light.png"
  >
  <img
    src="docs/assets/llmcompressor-icon-name-light.png"
    alt="LLM Compressor"
    width="420"
  >
</picture>

[![docs](https://img.shields.io/badge/docs-LLM--Compressor-blue)](https://docs.vllm.ai/projects/llm-compressor/en/latest/) [![PyPI](https://img.shields.io/pypi/v/llmcompressor.svg)](https://pypi.org/project/llmcompressor/)

</div>

`llmcompressor` is the fast, efficient, and easy-to-use library for optimizing models for deployment with vLLM, including:

* Comprehensive set of quantization algorithms and transforms for weight, activation, KV cache, and attention quantization
* Seamless integration with Hugging Face models and repositories
* Models saved in the `compressed-tensors` format, compatible with vLLM
* DDP and disk offloading support for compressing very large models with hardware efficiency

**✨ Read the announcement blog [here](https://neuralmagic.com/blog/llm-compressor-is-here-faster-inference-with-vllm/)! ✨**

<p align="center">
   <img alt="LLM Compressor Flow" src="https://github.com/user-attachments/assets/adf07594-6487-48ae-af62-d9555046d51b" width="80%" />
</p>

---

📊 Help us improve by taking our [1-minute user survey](https://red.ht/llm-compressor-user-survey)

💬 Join us on the [vLLM Community Slack](https://inviter.co/vllm-slack) and share your questions, thoughts, or ideas in:

- `#sig-quantization`
- `#llm-compressor`

---
## 🚀 What's New!

Big updates have landed in LLM Compressor! To get a more in-depth look, check out the [LLM Compressor overview](https://docs.google.com/presentation/d/1WNkYBKv_CsrYs69lb7bJKjh2dWt8U1HXUw7Gr4Wn3gE/edit?usp=sharing).

Since the v0.13.0 release, a number of meaningful improvements have landed:

* **Batched GPTQ quantization with a new Triton GPTQ kernel**: GPTQ now ships a Triton-based quantization kernel (~15x faster than the previous eager path) together with the ability to batch layers that share the same shape (up to ~1.67x per batch, roughly ~30x end-to-end on MoE workloads). Activation-order (act-order) calibration is supported, hessian offloading has been removed, and the remaining eager path was also sped up by 1.5-2x on its own.
* **Expanded MSE and iMatrix observers for FP4, with a new `fouroversix` default**: The MSE observer and the iMatrix observer gained a grid-search expansion factor that makes the search a strict superset of *fouroversix* (which chooses between the full `absmax` and `absmax * 1.5` scales for FP4 blocks). These observers outperform GPTQ for NVFP4 on average across our internal perplexity benchmarks.
* **Triton grid-search kernel for the MSE observer**: A Triton kernel now performs the MSE observer's scale grid search using buffered per-qparam patience and adaptive 512-value tiling. It reaches bitwise parity with the eager path when configured for full evaluation, supports INT, FP4, FP8, and FP16/BF16 (with E8M0 scales), and defaults `triton_error_buffer` to 100% for FP4 and 30% otherwise.

### Model highlights

The Red Hat AI team has been using LLM Compressor to produce a fresh batch of production-ready quantized checkpoints:

* **GLM-5.3 MXFP4**: An MXFP4 quantized checkpoint for [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3). The linear operators within the transformer blocks are quantized to MXFP4, while the MoE router, embeddings, DSA indexer, and output head are kept in their original precision to maintain accuracy recovery.
  - [RedHatAI/GLM-5.3-MXFP4](https://huggingface.co/RedHatAI/GLM-5.3-MXFP4)
  - [GLM-5.3 MXFP4 Example](examples/model_free_ptq/glm_5_3_mxfp4.py)
* **GLM-5.3-Flash NVFP4**: An NVFP4 quantized checkpoint for [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). The expert layers are quantized to NVFP4, while the MTP (multi-token prediction) layers are quantized to per-block FP8.
  - [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
  - [GLM-5.3-Flash NVFP4 Example](examples/quantizing_moe/glm53_flash_example.py)
* **Qwen3.8-Flash-Next NVFP4**: An NVFP4 quantized checkpoint for [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
  - [RedHatAI/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-Flash-Next-NVFP4)
* **Qwen3.8-27B INT4, NVFP4, and MXFP4**: 4-bit quantized checkpoints for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) across three formats — INT4, NVFP4, and MXFP4 — covering a range of hardware and accuracy trade-offs.
  - [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)
  - [RedHatAI/Qwen3.8-27B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4)
  - [RedHatAI/Qwen3.8-27B-MXFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4)
  - [Qwen3.8-27B INT4 Example](examples/quantization_w4a16/qwen3_8_gptq_awq_example.py)
  - [Qwen3.8-27B NVFP4 Example](examples/quantization_w4a4_fp4/qwen3_8_gptq_awq_example.py)
  - [Qwen3.8-27B MXFP4 Example](examples/quantization_w4a4_mxfp4/qwen3_8_gptq_awq_example.py)
* **Nemotron 3.5 Lightning FP8**: An FP8 quantized checkpoint for [Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16), created using GPTQ-based FP8 quantization.
  - [RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8)
  - [Nemotron 3.5 Lightning FP8 Example](examples/quantization_w8a8_fp8/nemotron_3_5_lightning_example.py)
* **Qwen3.8-2.4T-A95B NVFP4, NVFP4+FP8, and REAP+NVFP4**: NVFP4 and NVFP4+FP8 quantized checkpoints for [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B), along with `REAP-25` and `REAP-50` variants that combine [REAP](https://arxiv.org/pdf/2510.13999) expert pruning (25% and 50% of the least-salient experts pruned prior to quantization) with NVFP4, further reducing VRAM requirements while maintaining accuracy recovery.
  - [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4)
  - [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8)
  - [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-25](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-25)
  - [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-50](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-50)
  - [Qwen3.8-2.4T-A95B NVFP4+FP8 Example](examples/quantizing_moe/qwen_3_8_example.py)
  - [Qwen3.8 REAP + NVFP4 Example](examples/reap_expert_pruning/qwen38_example.py)
* **Muse-Glimmer-30B FP8, NVFP4, and INT4**: FP8, NVFP4, and INT4 checkpoints for [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), enabling single-GPU deployment of this multimodal model.
  - [RedHatAI/Muse-Glimmer-30B-FP8-block](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-FP8-block)
  - [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4)
  - [RedHatAI/Muse-Glimmer-30B-INT4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-INT4)
  - [Muse-Glimmer FP8_Block Example](examples/model_free_ptq/muse_glimmer_fp8_block.py)
* **Kimi-K3 NVFP4 and FP8**: NVFP4 and per-block FP8 quantized checkpoints for [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3). A native `KimiK3ForConditionalGeneration` model definition is now shipped in the library to support quantizing this architecture.
  - [RedHatAI/Kimi-K3-NVFP4](https://huggingface.co/RedHatAI/Kimi-K3-NVFP4)
  - [RedHatAI/Kimi-K3-FP8-BLOCK](https://huggingface.co/RedHatAI/Kimi-K3-FP8-BLOCK)
  - [Kimi-K3 Quantization Example](examples/quantizing_moe/kimi_k3_example.py)
* **Hy3 NVFP4+FP8**: A quantized checkpoint for [Hy3](https://huggingface.co/tencent/Hy3) combining NVFP4 quantization of the MoE layers with FP8 quantization of the attention layers, significantly reducing VRAM requirements while maintaining accuracy recovery.
  - [RedHatAI/Hy3-NVFP4-FP8](https://huggingface.co/RedHatAI/Hy3-NVFP4-FP8)
  - [Hy3 Quantization Example](examples/quantization_w4a4_fp4/hy3_example.py)
* **GLM-5.2 NVFP4+FP8**: Mixed-precision quantized checkpoints for [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), created with DDP + disk offloading in under 2 hours. NVFP4 quantization of the MoE layers and FP8 quantization of the attention layers reduces the model size by >70% while maintaining state-of-the-art accuracy recovery on GPQA.
  - [RedHatAI/GLM-5.2-NVFP4-FP8](https://huggingface.co/RedHatAI/GLM-5.2-NVFP4-FP8)
  - [RedHatAI/GLM-5.2-NVFP4](https://huggingface.co/RedHatAI/GLM-5.2-NVFP4)
  - [GLM-5.2 Example Script](examples/quantizing_moe/glm5_example.py)


### Supported Precisions and Types
* Activation Quantization: W8A8 (int8 and fp8), W4AFP8, Microscale (NVFP4, MXFP4, MXFP8)
* Mixed Precision: W4A16, W8A16, MXFP8A16, MXFP4A16, NVFP4A16
* Attention and KV Cache Quantization: FP8, NVFP4
* Low/Arbitrary-bit Quantization: WNA4, WNA8, WNA16

### Supported Algorithms
* Simple PTQ
* GPTQ
* AWQ
* SmoothQuant
* AutoRound
* Rotation-based (SpinQuant, QuIP)
* REAP expert pruning

### Quantizing your model, step-by-step

Please refer to our [step-by-step compression guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-model/) for detailed information about selecting quantization schemes, algorithms, and their use cases.

Additional information about LLM Compressor functionality is also available in our [User Guides](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/entrypoints/) and [FAQ](https://docs.vllm.ai/projects/llm-compressor/en/latest/faq/faq/).


## Installation

```bash
pip install llmcompressor
```

## Get Started

### End-to-End Examples

Applying quantization with `llmcompressor`:

### Weight and Activation Quantization
* [Activation quantization to `int8`](examples/quantization_w8a8_int8/README.md)
* [Activation quantization to `fp8`](examples/quantization_w8a8_fp8/README.md)
* [Activation quantization to MXFP8](examples/quantization_w8a8_mxfp8)
* [Activation quantization to `fp4` (NVFP4)](examples/quantization_w4a4_fp4)
* [Activation quantization to `fp4` (MXFP4)](examples/quantization_w4a4_mxfp4)
* [Activation quantization to `fp4` using AutoRound](examples/autoround/quantization_w4a4_fp4/README.md)
* [Activation quantization to `fp8` and weight quantization to `int4`](examples/quantization_w4a8_fp8)

### Weight Only Quantization
* [Weight only quantization to `fp4` (NVFP4 format)](examples/quantization_w4a16_fp4/nvfp4)
* [Weight only quantization to `fp4` (MXFP4 format)](examples/quantization_w4a16_fp4/mxfp4)
* [Weight only quantization to `int4` using GPTQ](examples/quantization_w4a16/README.md)
* [Weight only quantization to `int4` using AWQ](examples/awq/README.md)
* [Weight only quantization with AutoRound (`wNa16`)](examples/autoround/quantization_wNa16/README.md)

### Attention and KV Cache Quantization
* [KV Cache quantization to `fp8`](examples/quantization_kv_cache/README.md)
* [KV Cache quantization to `fp8` using per-head](examples/quantization_kv_cache/llama3_fp8_head_kv_example.py)
* [Attention quantization to `fp8`](examples/quantization_attention/README.md)
* [Attention quantization to `NVFP4` with SpinQuant (experimental)](experimental/attention/README.md)

### Architecture-Specific Quantization
* [Quantizing MoE LLMs](examples/quantizing_moe/README.md)
* [Quantizing Vision-Language Models](examples/multimodal_vision/README.md)
* [Quantizing Audio-Language Models](examples/multimodal_audio/README.md)

### Non-Uniform Quantization
* [Quantizing Models Non-uniformly](examples/quantization_non_uniform/README.md)

### Big Model Quantization Support
* [Quantizing large models with sequential onloading](examples/big_models_with_sequential_onloading/README.md)
* [Quantizing large models with disk offloading](examples/disk_offloading/README.md)

### Model-Free Definition Quantization
* [Quantizing models without a Hugging Face model definition](examples/model_free_ptq/README.md)

### DDP Quantization
* [Distributed data parallel quantization with GPTQ](examples/quantization_w4a16/llama3_ddp_example.py)


## Quick Tour
Let's quantize `Qwen3-30B-A3B` with FP8 weights and activations using the `Round-to-Nearest` algorithm.

Note that the model can be swapped for a local or remote HF-compatible checkpoint and the `recipe` may be changed to target different quantization algorithms or formats.

### Apply Quantization
Quantization is applied by selecting an algorithm and calling the `oneshot` API.

```python
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-30B-A3B"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to FP8 using RTN with block_size 128
#   * quantize the activations dynamically to FP8 during inference
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head", "re:.*mlp.gate$"],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-BLOCK"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
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
model = LLM("Qwen/Qwen3-30B-A3B-FP8-BLOCK")
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
