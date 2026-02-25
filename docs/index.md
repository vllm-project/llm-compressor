# What is LLM Compressor?

**LLM Compressor** is an easy-to-use library for optimizing large language models for deployment with vLLM. It provides a comprehensive toolkit for applying state-of-the-art compression algorithms to reduce model size, lower hardware requirements, and improve inference performance.

<p align="center">
   <img alt="LLM Compressor Flow" src="assets/llmcompressor-user-flows.png" width="100%" style="max-width: 100%;"/>
</p>

## Which challenges does LLM Compressor address?

Model optimization through quantization and pruning addresses the key challenges of deploying AI at scale:

| Challenge | How LLM Compressor helps |
|-----------|--------------------------|
| GPU and infrastructure costs | Reduces memory requirements by 50-75%, enabling deployment on fewer GPUs |
| Response latency | Reduces data movement overhead because quantized weights load faster |
| Request throughput | Utilizes lower-precision tensor cores for faster computation |
| Energy consumption | Smaller models consume less power during inference |

For more information, see [Why use LLM Compressor?](./steps/why-llmcompressor.md)

## New in this release

Review the [LLM Compressor v0.9.0 release notes](https://github.com/vllm-project/llm-compressor/releases/tag/0.9.0) for details about new features. Highlights include:

!!! info "Batched Calibration Support"
    LLM Compressor now supports calibration with batch sizes > 1. A new batch_size argument has been added to the dataset_arguments enabling the option to improve quantization speed. Default batch_size is currently set to 1

!!! info "New Model-Free PTQ Pathway"
    A new model-free PTQ pathway has been added to LLM Compressor, called model_free_ptq. This pathway allows you to quantize your model without the requirement of Hugging Face model definition and is especially useful in cases where oneshot may fail. This pathway is currently supported for data-free pathways only, such as FP8 quantization and was leveraged to quantize the Mistral Large 3 model. Additional examples have been added illustrating how LLM Compressor can be used for Kimi K2

!!! info "Extended KV Cache and Attention Quantization Support"
    LLM Compressor now supports attention quantization. KV Cache quantization, which previously only supported per-tensor scales, has been extended to support any quantization scheme including a new per-head quantization scheme. Support for these checkpoints is ongoing in vLLM and scripts to get started have been added to the [experimental](https://github.com/vllm-project/llm-compressor/tree/main/experimental) folder

!!! info "Generalized AWQ Support"
    The `AWQModifier` has been updated to support quantization schemes beyond W4A16 (e.g., W4AFp8). In particular, AWQ no longer constrains that the quantization config needs to have the same settings for group_size, symmetric, and num_bits for each config_group

!!! info "AutoRound Quantization Support"
    Added AutoRoundModifier for quantization using AutoRound, an advanced post-training algorithm that optimizes rounding and clipping ranges through sign-gradient descent. This approach combines the efficiency of post-training quantization with the adaptability of parameter tuning, delivering robust compression for large language models while maintaining strong performance

!!! info "Experimental MXFP4 Support"
    Models can now be quantized using an MXFP4 pre-set scheme. Examples can be found under the experimental folder. This pathway is still experimental as support and validation with vLLM is still a WIP.

## Supported algorithms and techniques

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **RTN** (Round-to-Nearest) | Fast baseline quantization | Quick compression with minimal setup |
| **GPTQ** | Weighted quantization with calibration | High-accuracy 4 and 8 bit weight quantization |
| **AWQ** | Activation-aware weight quantization | Preserves accuracy for important weights |
| **SmoothQuant** | Outlier handling for W8A8 | Improved activation quantization |
| **SparseGPT** | Pruning with quantization | 2:4 sparsity patterns |
| **SpinQuant** | Rotation-based transforms | Improved low-bit accuracy |
| **QuIP** | Incoherence processing | Advanced quantization preprocessing |
| **FP8 KV Cache** | KV cache quantization | Long context inference on Hopper-class and newer GPUs |
| **AutoRound** | Optimizes rounding and clipping ranges via sign-gradient descent | Broad compatibility |

## Supported quantization schemes

LLM Compressor supports applying multiple formats in a given model.

| Format | Targets | Compute Capability | Use Case |
|--------|---------|-------------------|----------|
| **W4A16/W8A16** | Weights | 8.0 (Ampere and up) | Optimize for latency on older hardware |
| **W8A8-INT8** | Weights and activations | 7.5 (Turing and up) | Balanced performance and compatibility |
| **W8A8-FP8** | Weights and activations | 8.9 (Hopper and up) | High throughput on modern GPUs |
| **NVFP4/MXFP4** | Weights and activations | 10.0 (Blackwell) | Maximum compression on latest hardware |
| **W4AFP8** | Weights and activations  | 8.9 (Hopper and up) | Low-bit weights with dynamic FP8 activations |
| **W4AINT8** | Weights and activations  | 7.5 (Turing and up) | Low-bit weights with dynamic INT8 activations |
| **2:4 Sparse** | Weights | 8.0 (Ampere and up) | Sparsity-accelerated inference |

!!! note
    Listed compute capability indicates the minimum architecture required for hardware acceleration.