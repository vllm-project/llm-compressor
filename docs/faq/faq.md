# Frequently Asked Questions

Below are the most frequently asked questions when using LLM Compressor. If you do not see your question here, please file an issue: [LLM Compressor Issues](https://github.com/vllm-project/llm-compressor/issues).

**1. Why doesn't my model run any faster after I compress it?**

This is usually the case when loading your model through transformers, not an inference server that supports models in the compressed-tensors format. Loading the model through transformers does not provide an inference benefit, as forward passes of the model are done with the model decompressed. There is no support for optimized compression inference during runtime. Instead, the model should be run in vLLM or another inference server that supports optimized inference for the quantized models.

**2. Are models compressed using LLM Compressor supported with SGlang?**

There is minimal support for compressed-tensors models in sglang, but it is not maintained nor tested by our team. Much of the integration relies on vLLM. For the most up-to-date and tested integration, vLLM is recommended.

**3. How do I choose the right quantization scheme?**

This involves understanding your hardware availability and inference requirements. Refer to [Compression Schemes Guide](../guides/compression_schemes.md).

**4. What are the memory requirements for compression?**

Refer to [Memory Requirements for LLM Compressor](../guides/memory.md).

**5. Which model layers should be quantized?**

Typically, all linear layers are quantized except the `lm_head` layer. This is because the `lm_head` layer is the last layer of the model and sensitive to quantization, which will impact the model's accuracy. For example, [this code snippet shows how to ignore the lm_head layer](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/llama3_example.py#L18).

Mixture of Expert (MoE) models, due to their advanced architecture and some components such as gate and routing layers, are sensitive to quantization as well. For example, [this code snippet shows how to ignore the gates](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantizing_moe/qwen_example.py#L60).

Multimodal models (e.g., vision-language models)pair a language model with another component for image, audio, or video input as well as text. In these cases, the non-textual component is excluded from quantization, as it generally has fewer parameters and is more sensitive.

For more information, see [Quantizing Multimodal Audio Models](https://github.com/vllm-project/llm-compressor/tree/main/examples/multimodal_audio) and [Quantizing Multimodal Vision-Language Models](https://github.com/vllm-project/llm-compressor/tree/main/examples/multimodal_vision).

**6. What environment should be used for installing LLM Compressor?**

 vLLM and LLM Compressor should be used in separate environments as they may have dependency mismatches.

**7. Does LLM Compressor have multi-GPU support?**

LLM Compressor enables the compression of large models through sequential onloading, whereby layers of the model are jointly onloaded to a single GPU, optimized, then offloaded back to the CPU. Consequently, in most cases, only one GPU is used at a time.

In cases where no calibration data is needed, the model is dispatched to all GPUs, although only one GPU is used at a time for compression.

Multi-GPU parallel optimization is currently in development and being tracked in this [issue](https://github.com/vllm-project/llm-compressor/issues/1809).

**8. Where can I learn more about LLM Compressor?**

There are multiple videos on YouTube:
[LLM Compressor deep dive + walkthrough](https://www.youtube.com/watch?v=caLYSZMVQ1c)
[vLLM Office Hours #23 - Deep Dive Into the LLM Compressor](https://www.youtube.com/watch?v=GrhuqQDmBk8)
[vLLM Office Hours #31 - vLLM and LLM Compresor Update](https://www.youtube.com/watch?v=WVenRmF4dPY)
[Optimizing vLLM Performance through Quantization|Ray Summit 2024](https://www.youtube.com/watch?v=G1WNlLxPLSE)