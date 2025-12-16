# LLM Compressor 0.9.0 release notes

This LLM Compressor 0.9.0 release adds support for MXFP4 compression, provides expanded quantization support, model compatibility improvements, and many other performance, workflow, and code optimizations.
This release also adds experimental support for R3 transforms and removes support for Python 3.9.

<!-- https://issues.redhat.com/browse/INFERENG-1889 -->
<!-- https://issues.redhat.com/browse/INFERENG-1890 -->
## Added support for MXFP4 compression ✨

LLM Compressor and compressed-tensors now support MXFP4 quantization and calibration of MXFP4 scales.
To use MXFP4 quantization, quantize models with the new MXFP4 preset scheme.
The `MXFP4PackedCompressor` class compresses and saves the model, packing both weights and scales as uint8 integers.
MXFP4 quantization is experimental pending validation with vLLM. 

Two new fields, `scale_dtype` and `zp_dtype`, have been added to the `QuantizationArgs` class: 

- `scale_dtype`: When set to `None`, scales are saved using the default dense data type.
When specified, scales are compressed using the provided data type.
For example, NVFP4 saves scales as FP8, while MXFP4 saves them as uint8.
This data type is reflected in the model config.
- `zp_dtype`: Set to `None` for symmetric models.
For asymmetric models, this specifies the data type used to save zero-point values.

See the [quant_args.py#L157](https://github.com/vllm-project/compressed-tensors/blob/797d3019ef6867362796f412980547c74551f369/src/compressed_tensors/quantization/quant_args.py#L157) implementation for details.

<!-- https://issues.redhat.com/browse/INFERENG-2163 -->
## Added experimental R3 transforms support

LLM Compressor now has expermental support for applying transforms to attention in the style of the R3 rotation used in SpinQuant models.
R3 transforms can potentially increase accuracy recovery for extreme attention quantization.
R3 rotations are not yet supported in vLLM.

<!-- https://issues.redhat.com/browse/INFERENG-2164 -->
## Extended LLM Compressor activation quantization support

LLM Compressor now supports dynamic group and channel activation quantization for models.

<!-- https://issues.redhat.com/browse/INFERENG-2365 -->
## Added support for Qwen3 VL MoE calibration

An updated MoE calibration context that enables correct calibration of expert layers in Mixture-of-Experts models has been added.
See the [moe_context.py#L29](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/moe_context.py#L29) implementation for details.

The calibration context can be used to temporarily or permanently update MoE module definitions during calibration, ensuring all expert models receive data during forward passes. 
This enables quantization support for Qwen3 VL and Qwen3 MoE models by using data-dependent schemes such as NVFP4, W4A16, and static activation quantization.

<!-- https://issues.redhat.com/browse/INFERENG-2542 -->
<!-- https://issues.redhat.com/browse/INFERENG-2543 -->
## AWQModifier runtime and lifecycle implementation improvements

LLM Compressor now supports batched calibration for quantization and sparsification.
Pass the `batch_size` and `data_collator` arguments to the `oneshot` compression entrypoint for improved calibration throughput.
For built-in collation strategies, pass `"padding"` or `"truncation"` as string values for `data_collator`.

AWQ has been generalized to support more quantization types.
The previous implementation used one-off quantization logic that limited supported configurations.
By adopting existing LLM Compressor abstractions, the code is now simpler and supports new quantization schemes including INT8, FP8, and mixed schemes within a single model.

<!-- https://issues.redhat.com/browse/INFERENG-2661 -->
<!-- https://issues.redhat.com/browse/INFERENG-2662 -->
## Refactored Observers

Observers functionality has been refactored for simplicity, with several new observers introduced:

- `memoryless_minmax`: Computes min/max values in real time using dynamic quantization style. Recommended for PTQ weight quantization.
- `static_minmax`: Computes absolute min/max values across all observations.
Recommended for PTQ activation quantization.
- `memoryless_mse`: Computes optimal quantization parameters by minimizing MSE loss for each observation.
Recommended for PTQ weight quantization.

`static_minmax` is now the default for NVFP4.
Future releases will standardize on `memoryless_minmax` for weight quantization and `static_minmax` for activation quantization.

NVFP4 now uses static observers for activation quantization by default.

<!-- https://issues.redhat.com/browse/INFERENG-2911 -->
## Updated support for KV Cache and attention quantization

KV cache quantization has been refactored to address limitations in supported schemes and lifecycle-related bugs.
The implementation has moved to compressed tensors, with new support for attention quantization.

This refactoring enables the following updates:
- Quantization of KV cache and attention using any scheme, including the new per-head strategy.
- Running KV/attention quantization experiments within Hugging Face for research and accuracy validation.
- Full compatibility with attention rotation via transforms.

These changes lay the groundwork for future work in creating and researching advanced attention and KV-cache quantized models.

<!-- https://issues.redhat.com/browse/INFERENG-3135 -->
<!-- https://issues.redhat.com/browse/INFERENG-3136 -->
## Added model-free post-training quantization (PTQ)

Model-free post-training quantization (PTQ) enables quantization by directly operating on safetensors files.
This is particularly useful for models without a `transformers` model definition, such as some Mistral models.

See the [model_free_ptq](https://github.com/vllm-project/llm-compressor/tree/main/examples/model_free_ptq) usage examples for more information.

With PTQ, you can quantize models by directly acting on safetensors files.
This is useful for quantizing models that do not have a `transformers` model definition, such as some Mistral models.

<!-- https://issues.redhat.com/browse/INFERENG-3572 -->
## Added matched_module implementation for AWQ and SmoothQuant PTQ

AWQ and SmoothQuant PTQ implementations previously used an suboptimal mapping-matching logic.
SmoothQuant couldn't handle MoE models, AWQ had buggy skip-layer logic, and neither could support certain parent contexts.

The `match_module_set` helper has been updated to handle all necessary situations, and both techniques now use this shared helper.
This enables SmoothQuant and AWQ support for MoE models, improves code maintainability, and eliminates several potential sources of bugs.

## Breaking changes

<!-- https://issues.redhat.com/browse/INFERENG-2422 -->
### Removed all training support APIs and functionality

Training support has been removed from LLM Compressor.
The `finetune` entrypoint and distillation modifier are no longer available.

For training workflows, use the LLM Compressor Axolotl integration.
See the [quantization_2of4_sparse_w4a16](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_2of4_sparse_w4a16) examples for details.

<!-- https://issues.redhat.com/browse/INFERENG-2426 -->
### Removed support for Python 3.9

LLM Compressor 0.9.0 requires Python 3.10 or later.
LLM Compressor 0.8.0 is the last version to support Python 3.9.