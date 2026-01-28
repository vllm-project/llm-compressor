# Choosing the right compression scheme

Before selecting a compression algorithm, you should first determine what format and compression scheme best fits your hardware and deployment requirements.

!!! info
    The general workflow is: **Choose your model → Choose your compression scheme → Choose your compression algorithm**

A compression scheme defines the numerical format and precision used to represent model weights and activations.
The scheme you choose determines both the compression ratio and the hardware required for acceleration.

| Scheme | Precision | Targets | GPU | Min. Compute Capability | Use Case |
|--------|-----------|---------|-----|-------------------------|----------|
| **W4A16** | 4-bit weights, 16-bit activations | Weights only | Ampere | 8.0 | Memory reduction on older hardware |
| **W8A8-INT8** | 8-bit integer | Weights and activations | Turing | 7.5 | Broad compatibility |
| **W8A8-FP8** | 8-bit floating point | Weights and activations | Hopper | 8.9 | High throughput on modern GPUs |
| **NVFP4** | 4-bit NVIDIA floating point | Weights and activations | Blackwell | 10.0 | Maximum compression on latest hardware |
| **MXFP4** | 4-bit MX floating point | Weights and activations | Blackwell | 10.0 | Maximum compression on latest hardware |
| **W4AFP8** | 4-bit weights, FP8 activations | Mixed precision | Hopper | 8.9 | Low-bit weights with FP8 activations |
| **W4AINT8** | 4-bit weights, INT8 activations | Mixed precision | Turing | 7.5 | Low-bit weights with INT8 activations |

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
FP8 can be applied using any quantization algorithm (RTN, AWQ, GPTQ), giving you flexibility in the accuracy-speed tradeoff.

See [FP8 weight and activation quantization](/examples/quantization_w8a8_fp8/) for more information.

## FP4 quantization (NVFP4/MXFP4)

4-bit floating point formats provide maximum compression on Blackwell GPUs, with 4x reduction compared to FP16.
FP4 is best with RTN algorithms for fast quantization with good accuracy recovery.

- **NVFP4**: is NVIDIA's native 4-bit format with block-wise scaling
- **MXFP4**: Microscaling FP4 format for cross-platform compatibility

!!! note
    FP4 quantization benefits significantly from calibration data to achieve optimal accuracy.

## Next steps

Once you've selected your compression scheme, then:

- [Choose the right compression algorithm](choosing-algo.md)
- [Compress the model](compress.md)