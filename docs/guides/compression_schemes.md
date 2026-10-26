# Compression Schemes

Below is a summary of the most popular schemes supported through LLM Compressor and compressed-tensors.
A full list of supported schemes can be found [here](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py).

- [W8A8-FP8](#fp8_dynamic)
- [W8A8-Block](#fp8_block)
- [W8A8-INT8](#int8_w8a8)
- [W4A16 and W8A16](#w4a16-and-w8a16)
- [NVFP4](#nvfp4)
- [MXFP4](#mxfp4)
- [MXFP4xFP8](#mxfp4xfp8)
- [MXFP6](#mxfp6)
- [MXFP8](#mxfp8)

## PTQ Compression Schemes


### FP8_DYNAMIC
| Scheme       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| W8A8-FP8      | 8-bit floating point (FP8) quantization for weights and activations                           |
| Weights       | Compressed ~2× smaller using channel-wise quantization (per-channel or per-tensor scales)    |
| Activations   | Quantized to 8-bit using dynamic per-token or static per-tensor methods; most performant with channel-wise weights + dynamic per-token activations |
| Calibration   | No calibration dataset required if using RTN; activation quantization happens during inference on vLLM    |
| Use case      | Optimized for performance and compression, especially for server and batch inference         |

### FP8_BLOCK
| Scheme       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| W8A8-FP8_BLOCK | 8-bit floating point (FP8) quantization using block-wise compression for weights             |
| Weights       | Compressed in blocks (commonly 128×128 tiles)                                                |
| Activations   | Quantized using dynamic per-group (128) quantization                                         |
| Calibration   | No calibration dataset required if using RTN; activation quantization happens during inference on vLLM    |
| Use case      | Optimized for performance and compression during inference                                   |


### INT8_W8A8
| Scheme       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| W8A8-INT8     | 8-bit integer (INT8) quantization for weights and activations, providing ~2× smaller weights with 8-bit arithmetic operations |
| Weights       | Compressed using per-channel scales (the `W8A8` preset); per-group weights are also supported |
| Activations   | Quantized to 8-bit using dynamic or static methods; can also be asymmetric        |
| Calibration   | Requires calibration dataset if using GPTQ/AWQ for weight quantization and for static activation quantization |
| Use case      | Optimized for general performance and compression, especially for server, batch inference, and high-QPS or offline serving with vLLM |


### W4A16 and W8A16

| Feature       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| WNA16         | Quantizes weights to 4 or 8-bit integer precision, retaining activations in 16-bit FP16  |
| Weights       | Typically ~3.7× compressed on a per-group basis; supports asymmetric quantization |
| Activations   | Retained in 16-bit floating point (FP16)                                                    |
| Calibration   | Optimally compressed using non-RTN algorithms (GPTQ, AWQ) which require a dataset           |                                                 
| Use case      | Maximum compression for latency-sensitive applications with limited memory; useful speedups in low-QPS regimes; recommended for any GPU |

Presets exist for every integer weight width from 2 to 8 bits (`W2A16` through `W8A16`), plus `W4A16_ASYM` for asymmetric 4-bit weights.


### NVFP4
| Feature       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| NVFP4         | 4-bit floating point format introduced with NVIDIA Blackwell GPUs; maintains accuracy using high-precision scale encoding and two-level micro-block scaling |
| Weights       | Compressed using global scale per tensor + local quantization scales per group of 16 elements |
| Activations   | Quantized dynamically using per-group quantization (group_size=16)                           |
| Scales        | Stored in FP8 format (`torch.float8_e4m3fn`)                                                 |
| Calibration   | Requires a calibration dataset to calibrate activation global scales                         |
| Use case      | Full FP4 weight and activation acceleration on NVIDIA Blackwell (SM100 and SM120). On older GPUs vLLM falls back to weight-only Marlin kernels with 16-bit activations. Use `NVFP4A16` for weight-only quantization, which needs no calibration data when using round to nearest. |


### MXFP4
| Feature       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| MXFP4         | 4-bit Microscaling floating point format defined by the OCP MX specification                 |
| Weights       | Compressed using per-group quantization (group_size=32)                                      |
| Activations   | Quantized fully dynamically using per-group quantization (group_size=32)                     |
| Scales        | Stored in E8M0 exponent format (`uint8`), extracting the power-of-2 exponent from each float scale value |
| Calibration   | No calibration data required if using RTN                                                    |
| Use case      | FP4 activations are only accelerated on NVIDIA Blackwell. Hopper runs MXFP4 MoE experts on native kernels with 16-bit activations, and other layers fall back to weight-only Marlin kernels on Ampere and newer. Cross-platform compatible via the OCP MX spec. An alternative to NVFP4 when calibration data is not available, though RTN accuracy may be lower. `MXFP4A16` is the weight-only variant |


### MXFP4xFP8
| Feature       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| MXFP4xFP8     | Mixed-precision scheme pairing MXFP4 weights with FP8 activations                            |
| Weights       | Same layout as MXFP4: per-group quantization (group_size=32) with E8M0 scales                |
| Activations   | Quantized fully dynamically to FP8 (E4M3) using per-group quantization (group_size=128)      |
| Scales        | Weight scales stored in E8M0 exponent format (`uint8`); activation scales are computed at inference time |
| Calibration   | No calibration data required if using RTN                                                    |
| Use case      | MXFP4 memory savings with FP8 activation accuracy, typically applied to MoE experts alongside `FP8` dense layers. See the [Qwen3.6 MoE example](https://github.com/vllm-project/llm-compressor/blob/main/examples/model_free_ptq/qwen3.6_moe_mxfp4xfp8.py) for usage |

!!! note
    Activations are quantized at inference time, so the vLLM kernel decides the format that actually runs.

    - **MoE experts**: Hopper uses the declared FP8 activations. Datacenter Blackwell (SM100) prefers a FlashInfer kernel that quantizes expert activations to MXFP8 instead.
    - **Dense (non-MoE) layers**: when the Humming kernels are installed, as in a default vLLM install, these layers run with FP8 activations. On datacenter Blackwell that Humming FP8 kernel is currently slow. With Humming disabled (`VLLM_DISABLED_KERNELS=HummingMxFp4LinearKernel`), vLLM falls back to plain MXFP4 handling. On Blackwell that quantizes activations to FP4, which is faster but less accurate; on older GPUs it runs weight-only with 16-bit activations.
    - **No FP8-capable kernel**: activations stay in 16-bit.


### MXFP6
| Feature       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| MXFP6         | 6-bit (E2M3) Microscaling floating point format defined by the OCP MX specification; 25% smaller weights than MXFP8 |
| Weights       | Compressed using per-group quantization (group_size=32); four values packed into three bytes |
| Activations   | `MXFP6A8` quantizes activations fully dynamically to MXFP8 (group_size=32); `MXFP6A16` keeps them in 16-bit |
| Scales        | Stored in E8M0 exponent format (`uint8`), extracting the power-of-2 exponent from each float scale value |
| Calibration   | No calibration data required if using RTN                                                    |
| Use case      | Memory reduction between MXFP4 and MXFP8; can be mixed with MXFP8 layers in one model         |


### MXFP8
| Feature       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| MXFP8         | 8-bit Microscaling floating point format defined by the OCP MX specification; higher accuracy than MXFP4 at the cost of lower compression |
| Weights       | Compressed using per-group quantization (group_size=32)                                      |
| Activations   | Quantized fully dynamically using per-group quantization (group_size=32)                     |
| Scales        | Stored in E8M0 exponent format (`uint8`), extracting the power-of-2 exponent from each float scale value |
| Calibration   | No calibration data required if using RTN                                                    |
| Use case      | Full MXFP8 acceleration on NVIDIA Blackwell; on older GPUs vLLM falls back to weight-only Marlin kernels. Cross-platform compatible via the OCP MX spec. `MXFP8A16` is the weight-only variant |

!!! warning
    Sparse compression (including 2of4 sparsity) is no longer supported by LLM Compressor due to lack of hardware support and user interest. Please see https://github.com/vllm-project/vllm/pull/36799 for more information.
