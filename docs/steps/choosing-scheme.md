# Choosing the right compression scheme

Before selecting a compression algorithm, you should first determine what format and compression scheme best fits your hardware and deployment requirements.

!!! info
    The general workflow is: **Choose your model → Choose your compression scheme → Choose your compression algorithm**

A compression scheme defines the numerical format and precision used to represent model weights and activations.
The scheme you choose determines both the compression ratio and the hardware required for acceleration.

| Scheme | Precision | Targets | GPU | vLLM min. compute capability | Use case |
|--------|-----------|---------|-----|-------------------------|----------|
| **W4A16/W8A16** | 4 or 8 bit weights, 16-bit activations | Weights | Turing | 7.5 | Memory reduction on older hardware |
| **W8A8-INT8** | 8-bit integer | Weights and activations | Turing | 7.5 | High throughput on older hardware |
| **W8A8-FP8** | 8-bit floating point | Weights and activations | Lovelace | 8.9 | High throughput on modern GPUs |
| **NVFP4** | 4-bit NVIDIA floating point | Weights and activations | Blackwell | 10.0 | Maximum compression on latest hardware |
| **MXFP4** | 4-bit MX floating point | Weights and activations | Blackwell | 10.0 | Maximum compression on latest hardware |
| **W4AFP8** | 4-bit weights, FP8 activations | Weights and activations | Hopper | 9.0 | Low-bit weights with FP8 activations |
| **W4AINT8** | 4-bit weights, INT8 activations | Weights and activations | Arm | - | Low-bit weights with INT8 activations |

!!! tip
    For more information, see [Compression schemes](../guides/compression_schemes.md).

## Choosing the right compression scheme for your GPU hardware

Your GPU architecture determines what compression schemes can be hardware-accelerated. For example:

### NVIDIA Blackwell
- **Minimum compute capability**: 10.0
- **Recommended**: NVFP4 or MXFP4 for maximum compression
- **Alternative**: FP8 for balanced compression and speed

### NVIDIA Hopper
- **Minimum compute capability**: 8.9
- **Recommended**: W8A8-FP8 for maximum throughput
- **Alternative**: W4AFP8 for mixed-precision with good accuracy

### NVIDIA Ampere
- **Minimum compute capability**: 8.0
- **Recommended**: W4A16 for memory reduction
- **Alternative**: W8A8-INT8 for weight and activation quantization

### NVIDIA Turing
- **Minimum compute capability**: 7.5
- **Recommended**: W8A8-INT8
- **Alternative**: W4A16

## FP8 quantization

FP8 (8-bit floating point) provides an excellent balance between compression and accuracy on Hopper-class and newer GPUs.
FP8 can be applied using any quantization algorithm (RTN, AWQ, GPTQ), allowing you to choose the accuracy-performance tradeoff that best fits your use case.

See [FP8 weight and activation quantization](/examples/quantization_w8a8_fp8/) for more information.

## FP4 quantization (NVFP4/MXFP4)

4-bit floating point formats provide maximum compression on Blackwell GPUs, with 4x reduction compared to FP16.
FP4 can sometimes provide good results with RTN algorithms for fast quantization, but potentially improved recovery can be gained using GPTQ or AWQ.

- **NVFP4**: is NVIDIA's native 4-bit format with block-wise scaling
- **MXFP4**: Microscaling FP4 format for cross-platform compatibility

## Compression Formats

Each quantization scheme corresponds to a particular quantization or sparsity compressor, which dictates
how the weights, scales, zero-points and other parameters are saved to disk after being compressed.
These compressors live in the [compressed-tensors](https://github.com/vllm-project/compressed-tensors/tree/main/src/compressed_tensors/compressors) project where a list of [available compressors](https://github.com/vllm-project/compressed-tensors/tree/main/src/compressed_tensors/config/base.py#L26) can be found. The table summarizies the common compression schemes and their corresponding compressed-tensors compressor.

For models with multiple precisions (e.g FP4 and FP8), multiple compressors may be applied to groups of layers. These models have a global mixed-precision format indicated in their
config.json while a local format is indicated for each group of targeted layers.


| Quantization  | Sparsity | Quant Compressor     | Sparsity Compressor |
|---------------|----------|----------------------|---------------------|
| W8A8 - int    | None     | int_quantized        | Dense               |
| W8A8 - float  | None     | float_quantized      | Dense               |
| NVFP4A16 - float | None     | nvfp4_pack_quantized | Dense               |
| NVFP4 - float  | None     | nvfp4_pack_quantized | Dense               |
| MXFP4A16 - float | None     | mxfp4_pack_quantized | Dense               |
| MXFP4 - float  | None     | mxfp4_pack_quantized | Dense               |
| W4A16 - int   | None     | pack_quantized       | Dense               |
| W4AFP8 - int   | None     | pack_quantized       | Dense               |
| W4AInt8 - int   | None     | pack_quantized       | Dense               |
| W8A16 - int   | None     | pack_quantized       | Dense               |
| W8A16 - float | None     | naive_quantized      | Dense               |
| W8A8 - int    | 2:4      | int_quantized        | Sparse24            |
| W8A8 - float  | 2:4      | float_quantized      | Sparse24            |
| W8A16 - float | 2:4      | naive_quantized      | Dense               |


## Next steps

- [Choose the right compression algorithm](choosing-algo.md)
- [Choosing your dataset](./choosing-dataset.md)
- [Compress the model](compress.md)
- [Deploy with vLLM](deploy.md)