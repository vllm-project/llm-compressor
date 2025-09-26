# Non-uniform Quantization

In certain cases, it may be useful to combine quantization schemes of different precisions and/or strategies to achieve better recovery. For example, in some decoder-only models, the `down_proj` layer has shown greater sensitivity, and performance can be improved by quantizing this layer to int8 or fp8 instead of int4 or fp4. The examples in this folder illustrate several cases of non-uniform quantization.

## Mixed-Precision Quantization

We demonstrate mixed precision by quantizing models to both int8 and int4, and in a second example, to both fp4 (specifically, nvfp4) and fp8. In both cases, we use config groups to assign higher precision to the `down_proj` layer and lower precision to the remaining linear layers. For nvfp4 and fp8, we also apply two model compressors—`nvfp4-pack-quantized` and `float-quantized`. The resulting compressed model’s config.json shows `mixed-precision` as the value for `format`, indicating that the model has been compressed using multiple formats. The specific format applied to each set of layers is specified under each config group’s `format` key.

## Multiple Strategies

It may also be interesting to quantize a model with two different [quantization strategies](https://github.com/neuralmagic/compressed-tensors/blob/a2bfc03e9d52824ba5d6d2a50c8741dd9bccd5d3/src/compressed_tensors/quantization/quant_args.py#L93) such as group, channel, or per-tensor. [Here](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_non_uniform/quantization_fp8_multiple_strategies.py) we apply fp8 quantization where all the attention weights are quantized using the per-channel strategy, and all the mlp weights are quantized using per-tensor. This is accomplished through defining multiple config groups in the recipe. The produced model is compressed using the `float-quantized` compressor and can be directly run in vllm.

## Quantization with Multiple Quantization Modifiers

This section outlines how multiple quantization modifiers can be applied to the same model for mixed-precision quantization, for example applying AWQ W4A16 to a model's `self_attn` layers and GPTQ W8A8 to its `mlp` layers. This heterogeneous application of multiple modifiers comes in 2 flavors:

1. Run every modifier in a single, sequential pipeline, performing a single calibrated run. See `./quantization_multiple_modifiers.py` for an example.
2. Run each modifier in its own, independent pipeline, performing a calibrated run for each modifier. To run each modifier independently, run the example with the `--independent` flag set (`python ./quantization_multiple_modifiers.py --independent`).

This is an advanced usage of `llm-compressor` and an active area of research. Best practices will be provided in a future release, after further research and sensitivity analysis.