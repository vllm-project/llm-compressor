# Choosing the right compression algorithm

After you've [chosen a compression scheme](choosing-scheme.md), you're ready to choose an algorithm to apply that scheme to your model.

LLM Compressor supports multiple quantization, pruning, and transform-based compression algorithms for different use cases.

!!! info
    Selecting the right compression algorithm depends on your chosen quantization scheme, accuracy requirements, and compatibility between the model, hardware, and algorithm.
    LLM Compressor provides a range of algorithms, from simple round-to-nearest quantization to advanced transform-based methods; each suited to different deployment scenarios.

## Weight and activation quantization

Weight and activation quantization is best for maximum throughput on modern hardware:

| Algorithm | Best for | Description | Accuracy Recovery vs. Time |
|-----------|----------|-------------| ---------------------------|
| SmoothQuant | Balanced compression | Balances weight and activation quantization for outlier handling | Good accuracy recovery with minimal calibration time; composable with other methods |
| AWQ | General purpose | AWQ (Activation-aware Weight Quantization) uses a small calibration set to identify the most important weights based on activation statistics. It preserves accuracy by rescaling the weights most coupled to these activations. | High accuracy recovery but can be expensive to run |
| GPTQ | Broad compatibility | Established weight quantization with calibration.  Utilizes second-order layer-wise optimizations to prioritize important weights/activations and enables updates to remaining weights |  High accuracy recovery but can be expensive to run |
| AutoRound | Broad compatibility  | Optimizes rounding and clipping ranges via sign-gradient descent | High accuracy recovery but can be expensive to run |
| RTN | Simple, data-free quantization |Simple quantization technique that rounds each value to the nearest representable level in the target precision. | Provides moderate accuracy recovery in most scenarios with good recovery for FP8/FP4. Computationally cheap and fast to implement, making it suitable for real-time or resource-constrained environments | 


!!! note
    AWQ and GPTQ are typically used for weight-only quantization but can also be applied to weight and activation quantization workflows.

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

| Algorithm | Best for |
|----------|-----------|
| RTN | Fast and simple compression |
| GPTQ or AWQ | Better accuracy at 4-bit (Int4 or FP4)|
| SmoothQuant | Smooths outliers in activations by folding them into weights and vice versa, ensuring better accuracy for weight+activation quantization |
| SparseGPT | 2:4 sparsity patterns |
| SpinQuant or QuIP + GPTQ | Best low-bit accuracy |
| FP8 KV Cache | Target KV Cache or attention activations |


### Mixed-precision quantization for accuracy recovery

For advanced use cases, LLM Compressor supports applying different quantization schemes to different model layers.
For example, you can combine FP4 for most layers with FP8 for sensitive layers to optimize the accuracy-effort tradeoff.

Not all model layers respond equally to quantization, some are more sensitive and require higher precision to maintain accuracy.
LLM Compressor supports non-uniform quantization, allowing you to apply different quantization schemes to different model layers within a single compression run.

You can also combine different quantization algorithms for different model layers, for example, applying AWQ to some layers and GPTQ to others within a single model.

With LLM Compressor, you can:

- Quantize most layers with FP4 for maximum compression
- Preserve sensitive layers (for example, attention blocks or first/last layers) at FP8
- Assign precision selectively by module type or layer group

This approach delivers better accuracy than uniform low-bit quantization while achieving smaller model sizes than uniform high-precision schemes.
See [the non-uniform quantization examples](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_non_uniform) for further details.

## Next steps

- [Choosing your dataset](./choosing-dataset.md)
- [Compress your first model](compress.md)
- [Deploy with vLLM](deploy.md)
