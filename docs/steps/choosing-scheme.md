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
| **NVFP4** | 4-bit NVIDIA floating point | Weights and activations | Blackwell (SM100) | 10.0 | Maximum compression on latest hardware |
| **MXFP4** | 4-bit MX floating point | Weights and activations | Blackwell (SM100) | 10.0 | Maximum compression; cross-platform compatible via OCP MX spec |
| **MXFP8** | 8-bit MX floating point | Weights and activations | Blackwell (SM100) | 10.0 | High accuracy MX format; cross-platform compatible via OCP MX spec |
| **W4AFP8** | 4-bit weights, FP8 activations | Weights and activations | Hopper | 9.0 | Low-bit weights with FP8 activations |
| **W4AINT8** | 4-bit weights, INT8 activations | Weights and activations | Arm | - | Low-bit weights with INT8 activations |

!!! tip
    For more information, see [Compression schemes](../guides/compression_schemes.md).

## Choosing the right compression scheme for your GPU hardware

Your GPU architecture determines what compression schemes can be hardware-accelerated. For example:

### NVIDIA Blackwell
- **Minimum compute capability**: 10.0
- **Recommended**: NVFP4 or MXFP4 for maximum compression
- **Alternative**: MXFP8 or FP8 for balanced compression and speed

### NVIDIA Hopper
- **Minimum compute capability**: 9.0
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

- **W8A8-FP8**: FP8 format with per-channel or per-tensor weight scales and dynamic per-token activation quantization
- **MXFP8**: Microscaling FP8 format using per-group quantization (group_size=32) with E8M0 scales; fully dynamic activations with no calibration data required; supported on Blackwell (SM100) GPUs

See [FP8 weight and activation quantization](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w8a8_fp8) for more information.

## FP4 quantization (NVFP4/MXFP4)

4-bit floating point formats provide maximum compression on Blackwell GPUs, with 4x reduction compared to FP16.
FP4 can sometimes provide good results with RTN algorithms for fast quantization, but potentially improved recovery can be gained using GPTQ or AWQ.

- **NVFP4**: NVIDIA's native 4-bit format with two-level micro-block scaling; requires calibration data for activation global scales
- **MXFP4**: Microscaling FP4 format for cross-platform compatibility; per-group quantization (group_size=32) with E8M0 scales; no calibration data required if using RTN

### Model size matters as much as hardware

The table above answers *what your GPU can run*. It does not answer *what your
model can absorb*: accuracy cost at 4 bits grows sharply as parameter count
falls, because smaller models carry less redundancy to spend on quantization
error. Recovery measured with the same recipe and the same calibration set on
one machine, paired per-item against the BF16 source:

| Model size | NVFP4 (W4A4) outcome |
|---|---|
| ~30B | multiple-choice recovery within noise (−0.5 pt, not significant) |
| ~7B | significant losses on generated output: −2.4 pt knowledge, −10.2 pt instruction following |

Two practical consequences:

- **Below roughly 10B parameters, prefer weight-only (W4A16) over W4A4**, or
  step up to FP8. Re-running the same 7B with W4A16 recovered the knowledge
  and instruction-following losses entirely (−0.7 pt and −3.7 pt, neither
  significant) at the same size and speed.
- **The two halves of W4A4 fail differently.** Quantizing activations is what
  damaged instruction following and knowledge; quantizing weights is what
  damaged long-chain mathematical reasoning (−5.4 pt, unchanged whether
  activations were 4-bit or 16-bit). If your workload is arithmetic- or
  reasoning-heavy, weight-only does not rescue it — validate before shipping.

!!! tip
    These are one lab's measurements on one model family, offered as a
    starting prior rather than a rule. Whatever scheme you pick, validate the
    quantized model against its own BF16 source on your own workload.

## Compression Formats

Each quantization scheme corresponds to a particular compressor, which dictates
how the weights, scales, zero-points and other parameters are saved to disk after being compressed.
These compressors live in the [compressed-tensors](https://github.com/vllm-project/compressed-tensors/tree/main/src/compressed_tensors/compressors) project where a list of [available compressors](https://github.com/vllm-project/compressed-tensors/tree/main/src/compressed_tensors/config/base.py#L26) can be found. The table summarizies the common compression schemes and their corresponding compressed-tensors compressor.

For models with multiple precisions (e.g FP4 and FP8), multiple compressors may be applied to groups of layers. These models have a global mixed-precision format indicated in their
config.json while a local format is indicated for each group of targeted layers.


| Quantization  | Quant Compressor        |
|---------------|-------------------------|
| W8A8 - int    | int_quantized           |
| W8A8 - float  | float_quantized         |
| NVFP4A16 - float | nvfp4_pack_quantized |
| NVFP4 - float  | nvfp4_pack_quantized   |
| MXFP4A16 - float | mxfp4_pack_quantized |
| MXFP4 - float  | mxfp4_pack_quantized   |
| MXFP8A16 - float | mxfp8_pack_quantized |
| MXFP8 - float  | mxfp8_pack_quantized   |
| W4A16 - int   | pack_quantized          |
| W4AFP8 - int   | pack_quantized         |
| W4AInt8 - int   | pack_quantized        |
| W8A16 - int   | pack_quantized          |
| W8A16 - float | naive_quantized         |

!!! warning
    Sparse compression (including 2of4 sparsity) is no longer supported by LLM Compressor due lack of hardware support and user interest. Please see https://github.com/vllm-project/vllm/pull/36799 for more information.

## Next steps

- [Choose the right compression algorithm](choosing-algo.md)
- [Choosing your dataset](./choosing-dataset.md)
- [Compress the model](compress.md)
- [Deploy with vLLM](deploy.md)
