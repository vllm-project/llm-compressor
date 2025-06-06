# Home

!!! info "New Feature: Axolotl Sparse Finetuning Integration"
    Easily finetune sparse LLMs through our seamless integration with Axolotl.
    [Learn more](https://docs.axolotl.ai/docs/custom_integrations.html#llmcompressor).

!!! info "New Feature: AutoAWQ Integration"
    Perform low-bit weight-only quantization efficiently using AutoAWQ, now part of LLM Compressor. [Learn more](https://github.com/vllm-project/llm-compressor/pull/1177).

## <div style="display: flex; align-items: center;"><img alt="LLM Compressor Logo" src="assets/llmcompressor-icon.png" width="40" style="vertical-align: middle; margin-right: 10px" /> LLM Compressor</div>

<p align="center">
   <img alt="LLM Compressor Flow" src="assets/llmcompressor-user-flows.png" width="100%" style="max-width: 100%;"s  />
</p>

**LLM Compressor** is an easy-to-use library for optimizing large language models for deployment with vLLM, enabling up to **5X faster, cheaper inference**. It provides a comprehensive toolkit for:

- Applying a wide variety of compression algorithms, including weight and activation quantization, pruning, and more
- Seamlessly integrating with Hugging Face Transformers, Models, and Datasets 
- Using a `safetensors`-based file format for compressed model storage that is compatible with `vLLM`
- Supporting performant compression of large models via `accelerate`

## Key Features

- **Weight and Activation Quantization:** Reduce model size and improve inference performance for general and server-based applications with the latest research.
    - Supported Algorithms: GPTQ, AWQ, SmoothQuant, RTN
    - Supported Formats: INT W8A8, FP W8A8
- **Weight-Only Quantization:** Reduce model size and improve inference performance for latency sensitive applications with the latest research
    - Supported Algorithms: GPTQ, AWQ, RTN
    - Supported Formats: INT W4A16, INT W8A16
- **Weight Pruning:** Reduce model size and improve inference performance for all use cases with the latest research
    - Supported Algorithms: SparseGPT, Magnitude, Sparse Finetuning
    - Supported Formats: 2:4 (semi-structured), unstructured

## Key Sections

<div class="grid cards" markdown>

- :material-rocket-launch:{ .lg .middle } Getting Started

    ---

    Install LLM Compressor and learn how to apply your first optimization recipe.

    [:octicons-arrow-right-24: Getting started](./getting-started/)

- :material-book-open-variant:{ .lg .middle } Guides

    ---

    Detailed guides covering compression schemes, algorithms, and advanced usage patterns.

    [:octicons-arrow-right-24: Guides](./guides/)

- :material-flask:{ .lg .middle } Examples

    ---

    Step-by-step examples for different compression techniques and model types.

    [:octicons-arrow-right-24: Examples](./examples/)

- :material-tools:{ .lg .middle } Developer Resources

    ---

    Information for contributors and developers extending LLM Compressor.

    [:octicons-arrow-right-24: Developer Resources](./developer/)

</div>
