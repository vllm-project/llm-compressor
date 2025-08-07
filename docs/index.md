# About LLM Compressor

**LLM Compressor** is an easy-to-use library for optimizing large language models for deployment with vLLM, enabling up to **5X faster, cheaper inference**. It provides a comprehensive toolkit for:

- Applying a wide variety of compression algorithms, including weight and activation quantization, pruning, and more
- Seamlessly integrating with Hugging Face Transformers, Models, and Datasets 
- Using a `safetensors`-based file format for compressed model storage that is compatible with `vLLM`
- Supporting performant compression of large models via `accelerate`

## <div style="display: flex; align-items: center;"><img alt="LLM Compressor Logo" src="assets/llmcompressor-icon.png" width="40" style="vertical-align: middle; margin-right: 10px" /> LLM Compressor</div>

<p align="center">
   <img alt="LLM Compressor Flow" src="assets/llmcompressor-user-flows.png" width="100%" style="max-width: 100%;"/>
</p>

## Recent Updates

!!! info "Llama4 Quantization Support"
    Quantize a Llama4 model to [W4A16](examples/quantization_w4a16.md) or [NVFP4](examples/quantization_w4a16.md). The checkpoint produced can seamlessly run in vLLM.

!!! info "Large Model Support with Sequential Onloading"
    As of llm-compressor>=0.6.0, you can now quantize very large language models on a single GPU. Models are broken into disjoint layers which are then onloaded to the GPU one layer at a time. For more information on sequential onloading, see [Big Modeling with Sequential Onloading](examples/big_models_with_sequential_onloading.md) as well as the [DeepSeek-R1 Example](examples/quantizing_moe.md).

!!! info "Preliminary FP4 Quantization Support"
    Quantize weights and activations to FP4 and seamlessly run the compressed model in vLLM. Model weights and activations are quantized following the NVFP4 [configuration](https://github.com/neuralmagic/compressed-tensors/blob/f5dbfc336b9c9c361b9fe7ae085d5cb0673e56eb/src/compressed_tensors/quantization/quant_scheme.py#L104). See examples of [weight-only quantization](examples/quantization_w4a16_fp4.md) and [fp4 activation support](examples/quantization_w4a4_fp4.md). Support is currently preliminary and additional support will be added for MoEs.

!!! info "Updated AWQ Support"
    Improved support for MoEs with better handling of larger models

!!! info "Axolotl Sparse Finetuning Integration"
    Seamlessly finetune sparse LLMs with our Axolotl integration. Learn how to create [fast sparse open-source models with Axolotl and LLM Compressor](https://developers.redhat.com/articles/2025/06/17/axolotl-meets-llm-compressor-fast-sparse-open). See also the [Axolotl integration docs](https://docs.axolotl.ai/docs/custom_integrations.html#llmcompressor).

For more information, check out the [latest release on GitHub](https://github.com/vllm-project/llm-compressor/releases/latest).

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
