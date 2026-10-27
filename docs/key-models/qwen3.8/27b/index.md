# Qwen3.8-27B

Quantization examples for the Qwen3.8-27B vision-language model.

> **Note:** These examples require `transformers >= v5`, which can be installed with:
> ```bash
> uv pip install --upgrade transformers
> ```
> With this, the examples can run end-to-end.

## Examples

- [4-bit Example](4bit-example.md)

## Pre-quantized Checkpoints

| Model | Format | Hugging Face Link |
| :--- | :--- | :--- |
| Qwen3.8-27B | INT4 (W4A16) | [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) |
| Qwen3.8-27B | NVFP4 | [RedHatAI/Qwen3.8-27B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) |
| Qwen3.8-27B | MXFP4 | [RedHatAI/Qwen3.8-27B-MXFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4) |
