# Compression Schemes

Below is a summary of the most popular schemes supported through LLM Compressor and compressed-tensors.
A full list of supported schemes can be found [here](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py).

- [W8A8-FP8](#fp8_dynamic)
- [W8A8-Block](#fp8_block)
- [W8A8-INT8](#int8_w8a8)
- [W4A16 and W8A16](#w4a16-and-w8a16)
- [NVFP4](#nvfp4)
- [2:4 Semi-structured Sparsity](#semi-structured)
- [Unstructured Sparsity](#unstructured)

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
| Weights       | Compressed using per-channel, per group |
| Activations   | Quantized to 8-bit using dynamic or static methods; can also be asymmetric        |
| Calibration   | Requires calibration dataset if using GPTQ/AWQ for weight qwuantization and for static activation quantization |
| Use case      | Optimized for general performance and compression, especially for server, batch inference, and high-QPS or offline serving with vLLM |


### W4A16 and W8A16

| Feature       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| WNA16         | Quantizes weights to 4 or 8-bit integer precision, retaining activations in 16-bit FP16  |
| Weights       | Typically ~3.7× compressed on a per-group or per-channel basis; supports asymmetric quantization |
| Activations   | Retained in 16-bit floating point (FP16)                                                    |
| Calibration   | Optimally compressed using non-RTN algorithms (GPTQ, AWQ) which require a dataset           |                                                 
| Use case      | Maximum compression for latency-sensitive applications with limited memory; useful speedups in low-QPS regimes; recommended for any GPU |


### NVFP4
| Feature       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| NVFP4         | 4-bit floating point format introduced with NVIDIA Blackwell GPUs; maintains accuracy using high-precision scale encoding and two-level micro-block scaling |
| Weights       | Compressed using global scale per tensor + local quantization scales per group of 16 elements |
| Activations   | Quantized dynamically using per-group quantization (group_size=16)                                      |
| Calibration   | Requires a calibration dataset to calibrate activation global scales                                                            |
| Use case      | Supported on all NVIDIA Blackwell GPUs or later  

## Sparsification Compression Schemes

Sparsification reduces model complexity by pruning selected weight values to zero while retaining essential weights in a subset of parameters. Supported formats include:


### Semi-Structured
| Feature       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| 2:4 Semi-structured Sparsity | Uses semi-structured sparsity (SparseGPT), where 2 of every 4 contiguous weights are set to zero. |
| Weights       | 2:4 sparsity                                                                                |
| Activations   | N/A                                                                                          |
| Calibration   | Requires a calibration dataset                                                              |
| Use case      | Fine-grained sparsity for compression and speedups           |



### Unstructured
| Feature       | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| Unstructured Sparsity | Zeros out individual weights without a regular pattern, removing weights wherever they contribute least. Produces a fine-grained sparse matrix. |
| Weights       | Sparsified individually (no structure)                                                     |
| Activations   | N/A                                                                  |
| Calibration   | Does not require a calibration dataset                                                    |
| Use case      | Fine-grained sparsity for compression and speedups                                         |
