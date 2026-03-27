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

## 🚀 Installation

Install `llmcompressor` via pip:

```bash
pip install llmcompressor
```

For the latest features, install from source:

```bash
pip install git+https://github.com/vllm-project/llm-compressor.git
```

---

## 🚀 What's New!

Big updates have landed in LLM Compressor! To get a more in-depth look, check out the [LLM Compressor overview](https://docs.google.com/presentation/d/1WNkYBKv_CsrYs69lb7bJKjh2dWt8U1HXUw7Gr4Wn3gE/edit?usp=sharing).

Some of the exciting new features include:

* **Qwen3.5 Support**: Qwen 3.5 can now be quantized using LLM Compressor. You will need to update your local transformers version using `uv pip install --upgrade transformers` and install LLM Compressor from source if using `<0.11`. Once updated, you should be able to run examples for the [MoE](examples/quantization_w4a4_fp4/qwen3_5_example.py) and [non-MoE](examples/quantization_w4a4_fp4/qwen3_5_example.py) variants of Qwen 3.5 end-to-end. For models quantized and published by the RedHat team, consider using the [NVFP4](https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-NVFP4) and FP8 checkpoints for [Qwen3.5-122B](https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic) and [Qwen3.5-397B](https://huggingface.co/RedHatAI/Qwen3.5-397B-A17B-FP8-dynamic).
* **Updated offloading and model loading support**: Loading transformers models that are offloaded to disk and/or offloaded across distributed process ranks is now supported. Disk offloading allows users to load and compress very large models which normally would not fit in CPU memory. Offloading functionality is no longer supported through accelerate but through model loading utilities added to compressed-tensors. For a full summary of updated loading and offloading functionality, for both single-process and distributed flows, see the [Big Models and Distributed Support guide](docs/guides/big_models_and_distributed/model_loading.md).
* **Distri