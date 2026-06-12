# Qwen3.5

Quantization examples for the Qwen3.5 family of models, including dense vision-language and sparse MoE variants.

> **Note:** These examples require `transformers >= v5`, which can be installed with:
> ```bash
> uv pip install --upgrade transformers
> ```
> With this, the examples can run end-to-end.

- [NVFP4A16 Vision-Language Example](nvfp4-vl-example.md)
- [NVFP4 MoE Example](nvfp4-moe-example.md)
- [W8A8 MoE Example](w8a8-moe-example.md)

## Pre-quantized Checkpoints

| Model | Format | Hugging Face Link |
| :--- | :--- | :--- |
| Qwen3.5-4B | FP8-dynamic | [RedHatAI/Qwen3.5-4B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3.5-4B-FP8-dynamic) |
| Qwen3.5-4B | W4A16 | [RedHatAI/Qwen3.5-4B-quantized.w4a16](https://huggingface.co/RedHatAI/Qwen3.5-4B-quantized.w4a16) |
| Qwen3.5-4B | W8A8 | [RedHatAI/Qwen3.5-4B-quantized.w8a8](https://huggingface.co/RedHatAI/Qwen3.5-4B-quantized.w8a8) |
| Qwen3.5-9B | FP8-dynamic | [RedHatAI/Qwen3.5-9B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3.5-9B-FP8-dynamic) |
| Qwen3.5-9B | W4A16 | [RedHatAI/Qwen3.5-9B-quantized.w4a16](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16) |
| Qwen3.5-9B | W8A8 | [RedHatAI/Qwen3.5-9B-quantized.w8a8](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w8a8) |
| Qwen3.5-35B-A3B | FP8-dynamic | [RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic) |
| Qwen3.5-122B-A10B | FP8-dynamic | [RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic) |
| Qwen3.5-122B-A10B | NVFP4 | [RedHatAI/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-NVFP4) |
| Qwen3.5-397B-A17B | FP8-dynamic | [RedHatAI/Qwen3.5-397B-A17B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3.5-397B-A17B-FP8-dynamic) |
