# Optimization Schemes

## PTQ
PTQ is performed to reduce the precision of quantizable weights (e.g., linear layers) to a lower bit-width. Supported formats are:

### [W4A16](../examples/quantization_w4a16/README.md)
- Uses GPTQ to compress weights to 4 bits. Requires calibration dataset.
- Optionally, [AWQ can also be leveraged for W4A16 quantization](../examples/awq/awq_one_shot.py)
- Useful speed ups in low QPS regimes with more weight compression. 
- Recommended for any GPUs types.

### [W8A8-INT8](../examples/quantization_w8a8_int8/README.md)
- Uses channel-wise quantization to compress weights to 8 bits using GPTQ, and uses dynamic per-token quantization to compress activations to 8 bits. Requires calibration dataset for weight quantization. Activation quantization is carried out during inference on vLLM.
- Useful for speed ups in high QPS regimes or offline serving on vLLM. 
- Recommended for NVIDIA GPUs with compute capability <8.9 (Ampere, Turing, Volta, Pascal, or older).

### [W8A8-FP8](../examples/quantization_w8a8_fp8/README.md)
- Uses channel-wise quantization to compress weights to 8 bits, and uses dynamic per-token quantization to compress activations to 8 bits. Does not require calibration dataset. Activation quantization is carried out during inference on vLLM.
- Useful for speed ups in high QPS regimes or offline serving on vLLM. 
- Recommended for NVIDIA GPUs with compute capability >=9.0 (Hopper and Blackwell).

## Sparsification
Sparsification reduces model complexity by pruning selected weight values to zero while retaining essential weights in a subset of parameters. Supported formats include:

### [2:4-Sparsity with FP8 Weight, FP8 Input Activation](../examples/sparse_2of4_quantization_fp8/README.md)
- Uses (1) semi-structured sparsity (SparseGPT), where, for every four contiguous weights in a tensor, two are set to zero. (2) Uses channel-wise quantization to compress weights to 8 bits and dynamic per-token quantization to compress activations to 8 bits.
- Useful for better inference than W8A8-fp8, with almost no drop in its evaluation score [blog](https://neuralmagic.com/blog/24-sparse-llama-fp8-sota-performance-for-nvidia-hopper-gpus/). Note: Small models may experience accuracy drops when the remaining non-zero weights are insufficient to recapitulate the original distribution.
- Recommended for compute capability >=9.0 (Hopper and Blackwell).