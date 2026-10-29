# Mistral-format model compression (experimental)

To quantize mistral models which do not have a huggingface model definition such as `mistralai/Devstral-Small-2505`, `mistralai/Magistral-Small-2506`, and `mistralai/mistral-large-3`, please use the [`model_free_ptq`](https://github.com/vllm-project/llm-compressor/tree/main/src/llmcompressor/entrypoints/model_free) entrypoint.