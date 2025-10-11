# Frequently Asked Questions

Below are the most frequently asked questions when using LLM Compressor. If you do not see your question here, please file an issue: [LLM Compressor Issues](https://github.com/vllm-project/llm-compressor/issues).

**1. Why doesn't my model run any faster after I compress it?**

This is usually the case when loading your model through transformers, not an inference server that supports models in the compressed-tensors format. Loading the model through transformers does not provide an inference benefit, as forward passes of the model are done with the model decompressed. There is no support for optimized compression inference during runtime. Instead, the model should be run in vLLM or another inference server that supports optimized inference for the quantized models.

**2. Do we support sglang?**

There is minimal support for compressed-tensors models in sglang, but it is not maintained nor tested by our team. Much of the integration relies on vLLM. For the most up-to-date and tested integration, vLLM is recommended.

**3. How do I select the appropriate strategy for compression?**

This involves understanding your hardware availability and inference requirements. Refer to [Compression Schemes Guide](../guides/compression_schemes.md).

**4. What are the memory requirements for compression?**

Refer to [Memory Requirements for LLM Compressor](compress.md#memory-requirements-for-llm-compressor).

**5. What layers should be quantized?**

Typically, all linear layers are quantized except the `lm_head` layer. This is because the `lm_head` layer is the last layer of the model and sensitive to quantization, which will impact the model's accuracy. For example, [this code snippet shows how to ignore the lm_head layer](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/llama3_example.py#L18).

Mixture of Expert (MoE) models, due to their advanced architecture and some components such as gate and routing layers, are sensitive to quantization as well. For example, [this code snippet shows how to ignore the gates](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantizing_moe/qwen_example.py#L60).

Multimodal models (e.g., vision-language models)pair a language model with another component for image, audio, or video input as well as text. In these cases, the non-textual component is excluded from quantization, as it generally has fewer parameters and is more sensitive.

For more information, see [Quantizing Multimodal Audio Models](https://github.com/vllm-project/llm-compressor/tree/main/examples/multimodal_audio) and [Quantizing Multimodal Vision-Language Models](https://github.com/vllm-project/llm-compressor/tree/main/examples/multimodal_vision).

**6. What environment should be used for installing LLM Compressor?**

 vLLM and LLM Compressor should be used in separate environments as they may have dependency mismatches.

**7. Does LLM Compressor have multi-GPU support?**

LLM Compressor handles all GPU movement for you. For data-free pathways, we leverage all available GPUs and offload anything that doesn't fit onto the allocated GPUs. If you are using pathways that require data, we sequentially onload model layers onto a single GPU. This is the case for LLM Compressor 0.6-0.8.
