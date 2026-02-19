# Select a Model

The first step in the compression workflow is selecting a compatible model. LLM Compressor supports a wide range of architectures including decoder-only language models (such as Llama3), multi-modal vision-language models (such as Qwen3 VL), and Mixture-of-Experts (MoE) models (such as Llama4, Kimi-K2, and Qwen3 VL MoE variants).

LLM Compressor integrates seamlessly with the Hugging Face ecosystem, allowing you to directly reference any compatible model from the Hugging Face Model Hub using its model identifier (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`). The library will automatically download and load the model, making Hugging Face the recommended starting point for most users. Alternatively, you can compress locally-stored models by providing a path to the model directory containing the model weights and configuration files. 

The following model architectures are fully supported in LLM Compressor:

| Model type | Notes |
|------------|-------|
| Standard language models |  Llama, Mistral, Qwen, and more |
| Multimodal/Vision models | Vision-language models |
| Mixture of Experts (MoE) models | DeepSeek, Qwen MoE, Mistral |
| Large multi-GPU models | Multi-GPU and CPU offloading support |

## Next Steps

- [Choosing the right compression scheme](choosing-scheme.md)
- [Choosing the right quantization, sparsity, and transform-based algorithms](choosing-algo.md)
- [Choosing your dataset](./choosing-dataset.md)
- [Compress your first model](compress.md)
- [Deploy with vLLM](deploy.md)
