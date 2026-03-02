# Why use LLM Compressor?

As AI models continue to grow in size and capability, deploying them efficiently becomes increasingly challenging.
LLM Compressor addresses these challenges through state-of-the-art quantization and pruning techniques.
Models produced by LLM Compressor have seamless integration with vLLM for efficient deployment.

**Advantages of using LLM Compressor**

| Benefit | Description |
|--------|---------|
| Reduced hardware costs | Deploy on fewer GPUs with 50-75% memory reduction |
| Improved inference speed | Lower latency and higher throughput through optimized kernels |
| Maintain accuracy | State-of-the-art algorithms preserve model quality |
| Broad model support | Works with standard LLMs, multimodal, and MoE architectures |
| Production-ready output | Direct integration with vLLM for deployment |
| Flexible algorithms | Choose the right technique for your hardware and accuracy needs |

The core challenge in LLM optimization is managing model size, inference speed, and accuracy.
LLM Compressor helps you find the optimal balance for your use case.
Model optimization through quantization and pruning directly addresses these challenges by reducing the computational and memory requirements of your models.

## Reduced hardware requirements means cheaper inference costs

Quantization reduces the precision of model weights and activations, dramatically reducing memory requirements.

Consider a 109B parameter BFloat16 baseline model at full precision requires ~220 GB (3 GPUs):

- Quantizing to INT8/FP8 halves the memory required to ~109 GB (2 GPUs)
- Quantizing to INT4/FP4 quarters the memory to ~55 GB (1 GPU)

## Improved performance

Optimization improves both latency and throughput:

- **Lower latency from data movement**: Quantized weights are faster to load from memory
- **Higher throughput via Tensor Cores**: Quantized activations enable faster computation using specialized hardware
- **Longer context support**: Reduced memory usage allows for larger KV caches

!!! important
    Research shows that properly applied quantization has minimal impact on model accuracy.
    Studies on models like DeepSeek-R1 show accuracy differences of less than 1% between full-precision and quantized versions.

### Quantizing the model reduces memory requirements

Quantization reduces model memory by representing weights and activations using a lower bit representation (for example, INT8 instead of FP16).
This allows models to use less storage and enables faster inference through specialized hardware tensor cores, providing the following benefits:

- Reduces memory footprint by 50-75%
- Enables deployment on memory-constrained hardware
- Leverages specialized tensor cores for faster computation

To quantize values, a scale and zero-point are computed to map the original high-precision values to a smaller range:

```
quantized_value = round(original_value / scale) + zero_point
```

### Pruning enables increased processing speed for hardware-accelerated compute

Pruning (or _sparsification_) zeros out certain model weight values in fixed patterns.
This can be done in specific patterns, such as **2:4 sparsity** where 2 out of every 4 values within a model weight tensor are set to 0. This has the following benefits:

- Enables more efficient computation
- Can be combined with quantization for additional gains
- Utilizes hardware acceleration available on modern GPUs

### Compressing the model reduces file size

Compression refers to saving the model in a reduced file size format with minimal impact to model accuracy.
LLM Compressor uses the `compressed-tensors` format, which is compatible with vLLM and Hugging Face.

## Common use cases for LLM Compressor

LLM Compressor supports a variety of optimization workflows depending on your deployment constraints and performance goals.

| Use Case | Scenario | Solution |
|----------|----------|----------|
| Deploying large models on limited hardware | Deploy a 70B parameter model on a single 80GB GPU | Apply INT4 quantization (W4A16) to reduce model size by 75%, enabling single-GPU deployment |
| Maximizing throughput for production serving | Serve high request volumes with minimal latency on modern NVIDIA hardware | Use FP8 quantization to leverage Hopper tensor cores for maximum throughput |
| Optimizing MoE models | Deploy a Mixture of Experts model like DeepSeek or Mixtral efficiently | Use NVFP4 quantization with calibration support designed for MoE architectures |

### Next Steps

- [Choosing your model](choosing-model.md)
- [Choosing the right compression scheme](choosing-scheme.md)
- [Choosing the right quantization, sparsity, and transform-based algorithms](choosing-algo.md)
- [Choosing your dataset](./choosing-dataset.md)
- [Compress your first model](compress.md)
- [Deploy with vLLM](deploy.md)
