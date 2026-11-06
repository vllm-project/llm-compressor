# DeepSeek V4 Flash

Quantization examples for the DeepSeek V4 Flash model.

DeepSeek V4 Flash uses a hybrid attention architecture with CSA, HCA, and mHC projections. The example below quantizes the MoE expert weights to NVFP4 while keeping the attention projections in FP8-Block.

## Examples

- [FP8-NVFP4 Example](fp8-nvfp4-example.md)

## Pre-quantized Checkpoints

- [RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8)
