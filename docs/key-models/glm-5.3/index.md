# GLM-5.3

Quantization examples for THUDM's GLM-5.3 MoE model.

The original [GLM-5.3 checkpoint](https://huggingface.co/zai-org/GLM-5.3) ships in an FP8-block quantized format. The examples below either dequantize it before applying a new scheme or re-quantize the already-compressed checkpoint.

## Examples

- [MXFP4 Example](mxfp4-example.md)
- [NVFP4 Layerwise Example](nvfp4-layerwise-example.md)

## Pre-quantized Checkpoints

| Model | Format | Hugging Face Link |
| :--- | :--- | :--- |
| GLM-5.3 | MXFP4 | [RedHatAI/GLM-5.3-MXFP4](https://huggingface.co/RedHatAI/GLM-5.3-MXFP4) |
| GLM-5.3 | NVFP4 | [RedHatAI/GLM-5.3-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-NVFP4) |
| GLM-5.3 | MXFP4xFP8 | [RedHatAI/GLM-5.3-MXFP4xFP8](https://huggingface.co/RedHatAI/GLM-5.3-MXFP4xFP8) |
