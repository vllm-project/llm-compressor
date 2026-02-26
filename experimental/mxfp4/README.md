# MXFP4 Quantization

vLLM currently supports MXFP4A16 quantization i.e weight-only quantization. Examples for this can be found [here](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w4a16_fp4/mxfp4).

However, you can still generate MXFP4 models through LLM Compressor. These models have fully dynamic activations and this pathway has not yet been enabled for compressed-tensors models in vLLM.