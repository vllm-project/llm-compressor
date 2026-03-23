# Qwen3.5

Quantization examples for the Qwen3.5 family of models, including dense vision-language and sparse MoE variants.

> **Note:** These examples require `transformers >= v5`, which can be installed with:
> ```bash
> uv pip install --upgrade transformers
> ```
> With this, the examples can run end-to-end on `main`. You may also need to update the version of `transformers` in your vLLM environment in order for the tokenizer to be properly applied.

- [NVFP4A16 Vision-Language Example](nvfp4-vl-example.md)
- [NVFP4 MoE Example](nvfp4-moe-example.md)
