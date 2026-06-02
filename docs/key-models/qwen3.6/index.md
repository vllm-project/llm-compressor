# Qwen3.6

Quantization examples for the Qwen3.6-35B-A3B sparse MoE model.

> **Note:** These examples require `transformers >= v5`, which can be installed with:
> ```bash
> uv pip install --upgrade transformers
> ```
> With this, the examples can run end-to-end.

- [NVFP4 MoE Example](nvfp4-moe-example.md)

Qwen3.6 shares the Qwen3.5 MoE architecture. For W8A8 and other schemes, use the [Qwen3.5 examples](../qwen3.5/index.md) and set `MODEL_ID` to `Qwen/Qwen3.6-35B-A3B`.

## Pre-quantized Checkpoints

- [RedHatAI/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4)
- [RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic)
