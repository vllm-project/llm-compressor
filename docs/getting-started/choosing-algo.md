# Choosing the right quantization, sparsity, and transform-based algorithms

LLM Compressor supports multiple quantization, sparsity, and transform-based algorithms for different use cases.

!!! info
    Selecting the right compression algorithm depends on your hardware, performance requirements, and acceptable accuracy tradeoffs.
    LLM Compressor provides a range of algorithms, from simple round-to-nearest quantization to advanced transform-based methods; each suited to different deployment scenarios.

## Weight-only quantization

Weight-only quantization is best for reducing model size when targeting memory-constrained hardware.

| Algorithm | Best for | Description |
|-----------|----------|-------------|
| AWQ | General purpose | Activation-aware weight quantization that preserves important weights |
| GPTQ | Broad compatibility | Established weight quantization with calibration |
| RTN | Quick baseline | Fast round-to-nearest quantization |

## Weight and activation quantization

Weight and activation quantization is best for maximum throughput on modern hardware:

| Algorithm | Best for | Description |
|-----------|----------|-------------|
| FP8 | Maximum speed | 8-bit floating point for NVIDIA Hopper+ GPUs |
| SmoothQuant | Balanced compression | Balances weight and activation quantization for outlier handling |

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
| Maximum throughput (Hopper+) | FP8 |
| Balanced weight/activation | SmoothQuant |
| 2:4 sparsity patterns | SparseGPT |
| Best low-bit accuracy | SpinQuant or QuIP + GPTQ |

## Supported model types

The following model architectures are fully supported in LLM Compressor:

| Model Type | Notes |
|------------|---------|
| Standard language models |  Llama, Mistral, Qwen, and more |
| Multimodal/Vision models | Vision-language models |
| Mixture of Experts (MoE) models | DeepSeek, Mixtral with NVFP4 calibration |
| Large multi-GPU models | CPU offloading via Hugging Face accelerate |

### Mixed-precision quantization for accuracy recovery

For advanced use cases, LLM Compressor supports applying different quantization schemes to different model components.
For example, you can combine NVFP4 for most layers with FP8 for sensitive layers to optimize the accuracy-performance tradeoff.

Not all model layers respond equally to quantization, some are more sensitive and require higher precision to maintain accuracy.
LLM Compressor supports non-uniform quantization, allowing you to apply different quantization schemes to different model components within a single compression run.

With LLM Compressor, you can:

- Quantize most layers with NVFP4 for maximum compression
- Preserve sensitive layers (for example, attention blocks or first/last layers) at FP8
- Assign precision selectively by module type or layer group

This approach delivers better accuracy than uniform low-bit quantization while achieving smaller model sizes than uniform high-precision schemes.
