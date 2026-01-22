# What is LLM Compressor?

**LLM Compressor** is an easy-to-use library for optimizing large language models for deployment with vLLM. It provides a comprehensive toolkit for applying state-of-the-art compression algorithms to reduce model size, lower hardware requirements, and improve inference performance.

<p align="center">
   <img alt="LLM Compressor Flow" src="assets/llmcompressor-user-flows.png" width="100%" style="max-width: 100%;"/>
</p>

## What challenges does LLM Compressor address?

Model optimization through quantization and sparsification addresses the key challenges of deploying AI at scale:

| Challenge | How LLM Compressor helps |
|-----------|--------------------------|
| GPU and infrastructure costs | Reduces memory requirements by 50-75%, enabling deployment on fewer GPUs |
| Response latency | Reduces data movement overhead because quantized weights load faster |
| Request throughput | Utilizes lower-precision tensor cores for faster computation |
| Energy consumption | Smaller models consume less power during inference |

For more information, see [Why use LLM Compressor?](./getting-started/why-llmcompressor.md)

## New in this release

Review the [LLM Compressor v0.8.0 release notes](https://github.com/vllm-project/llm-compressor/releases/tag/0.8.0) for details about new features. Highlights include:

!!! info "Support for multiple modifiers in oneshot compression runs"
    LLM Compressor now supports using multiple modifiers in oneshot compression runs such as applying both AWQ and GPTQ in a single model. 

    Using multiple modifiers is an advanced usage of LLM Compressor and an active area of research. See [Non-uniform Quantization](/examples/quantization_non_uniform/) for more detail and example usage.

!!! info "Quantization and calibration support for Qwen3 models"
    Quantization and calibration support for Qwen3 Next models has been added to LLM Compressor.

    LLM Compressor now supports quantization for Qwen3 Next and Qwen3 VL MoE models. You can now use data-free pathways such as FP8 channel-wise and block-wise quantization. Pathways requiring data such W4A16 and NVFP4 are planned for a future release.

    Examples for NVFP4 and FP8 quantization have been added for the Qwen3-Next-80B-A3B-Instruct model. 

    For the Qwen3 VL MoE model, support has been added for the data-free pathway. The data-free pathway applies FP8 quantization, for example, channel-wise and block-wise quantization. 

    **NOTE**: These models are not supported in tranformers<=4.56.2. You may need to install transformers from source.

!!! info "Transforms support for non-full-size rotation sizes"
    You can now set a `transform_block_size` field in the Transform-based modifier classes `SpinQuantModifier` and `QuIPModifier`. You can configure transforms of variable size with this field, and you don't need to restrict hadamards to match the size of the weight.

## Recent updates

!!! info "QuIP and SpinQuant-style Transforms" 
    The newly added [`QuIPModifier` and `SpinQuantModifier`](/examples/transform) transforms allow you to quantize models after injecting hadamard weights into the computation graph, reducing quantization error and greatly improving accuracy recovery for low bit-weight and activation quantization.

!!! info "DeepSeekV3-style Block Quantization Support" 
    Allows for more efficient compression of large language models without needing a calibration dataset. Quantize a Qwen3 model to [W8A8](/examples/quantization_w8a8_fp8/).

!!! info "FP4 Quantization - now with MoE and non-uniform support" 
    Quantize weights and activations to FP4 and seamlessly run the compressed model in vLLM. Model weights and activations are quantized following the [NVFP4 configuration](https://github.com/neuralmagic/compressed-tensors/blob/f5dbfc336b9c9c361b9fe7ae085d5cb0673e56eb/src/compressed_tensors/quantization/quant_scheme.py#L104). See examples of [FP4 activation support](/examples/quantization_w4a4_fp4/), [MoE support](/examples/quantization_w4a4_fp4/), and [Non-uniform quantization support](/examples/quantization_non_uniform/) where some layers are selectively quantized to FP8 for better recovery. You can also mix other quantization schemes, such as INT8 and INT4.

!!! info "Llama4 Quantization Support"
    Quantize a Llama4 model to [W4A16](/examples/quantization_w4a16/) or [NVFP4](/examples/quantization_w4a4_fp4/). The checkpoint produced can seamlessly run in vLLM.

For more information, check out the [latest release on GitHub](https://github.com/vllm-project/llm-compressor/releases/latest).

## Supported algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **RTN** (Round-to-Nearest) | Fast baseline quantization | Quick compression with minimal setup |
| **GPTQ** | Weighted quantization with calibration | High-accuracy 4-bit weight quantization |
| **AWQ** | Activation-aware weight quantization | Preserves accuracy for important weights |
| **SmoothQuant** | Outlier handling for W8A8 | Weight and activation quantization |
| **SparseGPT** | Pruning with quantization | 2:4 sparsity patterns |
| **SpinQuant** | Rotation-based transforms | Improved low-bit accuracy |
| **QuIP** | Incoherence processing | Advanced quantization preprocessing |

## Supported quantization formats

| Format | Targets | Compute Capability | Use Case |
|--------|---------|-------------------|----------|
| **W4A16** | Weights only | SM80 (Ampere+) | Optimize for latency on older hardware |
| **W8A8-INT8** | Weights + activations | SM75 (Turing+) | Balanced performance and compatibility |
| **W8A8-FP8** | Weights + activations | SM89 (Hopper+) | High throughput on modern GPUs |
| **W4A4-NVFP4** | Weights + activations | SM100 (Blackwell) | Maximum compression on latest hardware |
| **2:4 Sparse** | Weights | SM80 (Ampere+) | Sparsity-accelerated inference |
