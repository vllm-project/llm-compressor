# Qwen3.8-2.4T-A95B

Quantization examples for Qwen3.8's large sparse MoE model.

> **Note:** These examples require `transformers >= v5`, which can be installed with:
> ```bash
> uv pip install --upgrade transformers
> ```
> With this, the examples can run end-to-end.

## Examples

- [NVFP4 + FP8 Example](nvfp4-moe-example.md)
- [REAP + NVFP4 Example](reap-example.md)

## Pre-quantized Checkpoints

| Model | Format | Hugging Face Link |
| :--- | :--- | :--- |
| Qwen3.8-2.4T-A95B | NVFP4 | [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4) |
| Qwen3.8-2.4T-A95B | NVFP4 + FP8 | [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8) |
| Qwen3.8-2.4T-A95B | NVFP4 + REAP-25 | [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-25](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-25) |
| Qwen3.8-2.4T-A95B | NVFP4 + REAP-50 | [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-50](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-50) |
