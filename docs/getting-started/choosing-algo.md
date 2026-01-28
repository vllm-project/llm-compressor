# Choosing the right quantization, pruning, and transform-based algorithms

LLM Compressor supports multiple quantization, pruning, and transform-based algorithms for different use cases.

!!! info
    Selecting the right compression algorithm depends on your hardware, performance requirements, and acceptable accuracy tradeoffs.
    LLM Compressor provides a range of algorithms, from simple round-to-nearest quantization to advanced transform-based methods; each suited to different deployment scenarios.

## Weight-only quantization

Weight-only quantization is best for reducing model size when targeting memory-constrained hardware.

| Algorithm | Best for | Description |
|-----------|----------|-------------|
| AWQ | General purpose | Activation-aware weight quantization that preserves important weights |
| GPTQ | Broad compatibility | Established weight quantization with calibration |
| RTN | Quick baseline, FP4/FP8 | Fast round-to-nearest quantization. Good accuracy recovery for NVFP4, MXFP4, and FP8 formats |

## Weight and activation quantization

Weight and activation quantization is best for maximum throughput on modern hardware:

| Algorithm | Best for | Description |
|-----------|----------|-------------|
| RTN | FP8 quantization | Fast round-to-nearest quantization for FP8 weight and activation quantization |
| SmoothQuant | Balanced compression | Balances weight and activation quantization for outlier handling |

!!! info
    LLM Compressor also supports mixed-precision activation quantization, such as W4AFP8 and W4AINT8, allowing you to combine low-bit weights with higher-precision activations for improved accuracy.

## KV cache and attention quantization

KV cache quantization reduces memory usage for long context inference:

| Algorithm | Best for | Description |
|-----------|----------|-------------|
| FP8 KV Cache | Long context inference | Reduces KV cache memory footprint on Hopper-class and newer NVIDIA GPUs |

## Sparsity and transform-based algorithms

The following algorithms provide additional optimization beyond standard quantization, enabling further performance gains or improved accuracy recovery at low bit-widths.

| Algorithm | Best for | Description |
|-----------|----------|-------------|
| SparseGPT | Computational efficiency | Post-training structured pruning |
| SpinQuant | Low-bit quantization accuracy | Rotation-based transform that reduces quantization errors |
| QuIP | Research-grade quantization | Incoherence-based transforms for robust low-bit weight quantization |

## Compression algorithms

LLM Compressor provides multiple compression algorithms, each optimized for different goals.
Use the table below to select the algorithm that best matches your deployment requirements and hardware capabilities.

| Goal | Recommended Algorithm |
|------|----------------------|
| Fast and simple compression | RTN |
| Better accuracy at 4-bit | GPTQ or AWQ |
| Maximum throughput (Hopper and up) | FP8 |
| Maximum compression (Blackwell) | NVFP4/MXFP4 |
| Balanced weight/activation | SmoothQuant |
| 2:4 sparsity patterns | SparseGPT |
| Best low-bit accuracy | SpinQuant or QuIP + GPTQ |

!!! note
    FP8 quantization can be applied with any quantization algorithm, including RTN, AWQ, and GPTQ, allowing you to choose the accuracy-performance tradeoff that best fits your use case.

## Supported model types

The following model architectures are fully supported in LLM Compressor:

| Model Type | Notes |
|------------|---------|
| Standard language models |  Llama, Mistral, Qwen, and more |
| Multimodal/Vision models | Vision-language models |
| Mixture of Experts (MoE) models | DeepSeek, Qwen MoE, Mistral |
| Large multi-GPU models | CPU offloading via Hugging Face accelerate |

### Mixed-precision quantization for accuracy recovery

For advanced use cases, LLM Compressor supports applying different quantization schemes to different model layers.
For example, you can combine INT4 for most layers with FP8 for sensitive layers to optimize the accuracy-performance tradeoff.

Not all model layers respond equally to quantization, some are more sensitive and require higher precision to maintain accuracy.
LLM Compressor supports non-uniform quantization, allowing you to apply different quantization schemes to different model layers within a single compression run.

You can also combine different quantization algorithms for different model layers, for example, applying AWQ to some layers and GPTQ to others within a single model.

With LLM Compressor, you can:

- Quantize most layers with INT4 for maximum compression
- Preserve sensitive layers (for example, attention blocks or first/last layers) at FP8
- Assign precision selectively by module type or layer group

This approach delivers better accuracy than uniform low-bit quantization while achieving smaller model sizes than uniform high-precision schemes.
