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

Some of the exciting new features include:

* **Qwen3.8 NVFP4, FP8, and INT4 Quantized Checkpoints**: NVFP4 and FP8 quantized checkpoints for Qwen3.8-2.4T-A95B, along with an INT4 checkpoint for Qwen3.8-27B, have been created by the Red Hat AI team. Of particular note, `Qwen3.8-2.4T-A95B-NVFP4-REAP-25` combines [REAP expert pruning](#reap-expert-pruning-modifier) with NVFP4 quantization — 25% of the least-salient experts are pruned prior to quantization, further reducing VRAM requirements while maintaining accuracy recovery.
  - Models:
    - [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)
    - [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-25](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-25)
    - [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8)
    - [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4)
    - [RedHatAI/Qwen3.8-2.4T-A95B-FP8](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-FP8)
  - Examples:
    - [Qwen3.8-2.4T-A95B NVFP4+FP8 Example](examples/quantizing_moe/qwen_3_8_example.py)
    - [Qwen3.8-2.4T-A95B REAP + NVFP4 Example](examples/reap_expert_pruning/qwen38_example.py)
    - [Qwen3.8-27B INT4 Example](examples/quantization_w4a16/qwen3_8_gptq_awq_example.py)
* **Nemotron 3.5 Lightning FP8 Quantized Checkpoint**: An FP8 quantized checkpoint for [Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) has been created by the Red Hat AI team using GPTQ-based FP8 quantization.
  - [RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8)
  - [Nemotron 3.5 Lightning FP8 Example](examples/quantization_w8a8_fp8/nemotron_3_5_lightning_example.py)
* **Muse-Glimmer-30B FP8, NVFP4, and INT4 Quantized Checkpoints**: FP8, NVFP4, and INT4 checkpoints for [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) have been created by the Red Hat AI team, enabling single-GPU deployment of this multimodal model.
  - [RedHatAI/Muse-Glimmer-30B-FP8-block](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-FP8-block)
  - [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4)
  - [RedHatAI/Muse-Glimmer-30B-W4A16](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-W4A16)
  - [Muse-Glimmer FP8_Block Example](examples/model_free_ptq/muse_glimmer_fp8_block.py)
* **Kimi-K3 NVFP4 and FP8 Quantized Checkpoints**: NVFP4 and FP8 quantized checkpoints for Kimi-K3 have been created by the Red Hat AI team.
  - [RedHatAI/Kimi-K3-NVFP4](https://huggingface.co/RedHatAI/Kimi-K3-NVFP4)
  - [RedHatAI/Kimi-K3-FP8-BLOCK](https://huggingface.co/RedHatAI/Kimi-K3-FP8-BLOCK)
* **Hy3 NVFP4+FP8 Quantized Checkpoint**: A quantized checkpoint for [Hy3](https://huggingface.co/tencent/Hy3) has been created by the Red Hat AI team, combining NVFP4 quantization of MoE layers with FP8 quantization of attention layers to significantly reduce VRAM requirements while maintaining accuracy recovery.
  - [RedHatAI/Hy3-NVFP4-FP8](https://huggingface.co/RedHatAI/Hy3-NVFP4-FP8)
  - [Hy3 Quantization Example](examples/quantization_w4a4_fp4/hy3_example.py)
* **GLM-5.2 NVFP4+FP8 Example and Checkpoints**: Quantized checkpoints for [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) have been created by the Red Hat AI team using DDP + disk offloading in under 2 hours. The full precision model requires 1.6T of VRAM, but NVFP4 quantization of MoE layers and FP8 quantization of attention layers reduces the model size by >70% while maintaining state-of-the-art accuracy recovery on GPQA.
  - [RedHatAI/GLM-5.2-NVFP4-FP8](https://huggingface.co/RedHatAI/GLM-5.2-NVFP4-FP8)
  - [GLM-5.2 Example Script](examples/quantizing_moe/glm5_example.py)
<a id="reap-expert-pruning-modifier"></a>

* **REAP Expert Pruning Modifier**: [REAP](https://arxiv.org/pdf/2510.13999) reduces the VRAM requirements to run Mixture-of-Experts models by structurally removing less-relevant experts in each layer. With relevancy proxied by a saliency metric calculated from calibration forward pass data, REAP achieves a desired expert sparsity (set by the user) while aiming to minimize the impact of the pruned experts. The modifier implementation is in [`modifiers/pruning/reap`](src/llmcompressor/modifiers/pruning/reap) and can be used as a template for implementing other expert pruning algorithms. Examples and additional documentation can be found below:
  - [REAP Pruning README](examples/reap_expert_pruning/README.md)
  - [REAP Prune Qwen/Qwen3-30B-A3B-Instruct-2507 to 25% Sparsity](examples/reap_expert_pruning/reap_qwen3_30b.py)
  - [REAP Prune moonshotai/Moonlight-16B-A3B-Instruct to 25% Sparsity](examples/reap_expert_pruning/reap_moonlight_16b.py)



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
