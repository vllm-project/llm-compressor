# LLM Compressor Frequently Asked Questions

Below are the most frequently asked questions when using LLM Compressor. If you do not see your question here, please use this page to ask your question: [https://github.com/vllm-project/llm-compressor/issues](https://github.com/vllm-project/llm-compressor/issues).

**1. Why doesn't my model run any faster after I compress it?**

This is usually the case when loading your model through transformers, not an inference server that supports models in the compressed-tensors format. Loading the model through transformers does not provide an inference benefit, as forward passes of the model are done with the model decompressed. There is no support for optimized compression inference during runtime. Instead, the model should be run in vLLM or another inference server that supports optimized inference for the quantized models.

**2. Do we support sglang?**

There is minimal support for compressed-tensors models in sglang, but it is not maintained nor tested by our team. Much of the integration relies on vLLM. For the most up-to-date and tested integration, vLLM is recommended.

**3. How do I select the appropriate strategy for compression?**

This involves understanding your hardware availability and inference requirements. Refer to [https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/compression_schemes/](Compression Schemes Guide).

**4. How much memory or time will xyz algorithm take with my model?**

Refer to [https://docs.vllm.ai/projects/llm-compressor/en/latest/getting-started/compress/#memory-requirements-for-llm-compressor](Memory Requirements for LLM Compressor).

**5. What are the memory requirements?**

Refer to [https://docs.vllm.ai/projects/llm-compressor/en/latest/getting-started/compress/#memory-requirements-for-llm-compressor](Memory Requirements for LLM Compressor).

**6. What layers should be quantized?**

All linear layers go through basic quantization except the `lm_head` layer. This is because the `lm_head` layer is the last layer of the model and sensitive to quantization, which will impact the model's accuracy. For example, [https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/llama3_example.py#L18](here) is a code snippet of how to ignore the lm_layer.

Mixture of Expert (MoE) models, due to their advanced architecture and some components such as gate and routing layers, are sensitive to quantization as well. For example, [this code snippet shows how to ignore the gates](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantizing_moe/qwen_example.py#L60).
