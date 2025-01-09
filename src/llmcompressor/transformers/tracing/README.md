# Model Tracing Guide #
This guide explains the concepts of tracing as they relate to LLM Compressor and how to
modify your model or configuration to support recipes which require using the 
[Sequential Pipeline](/src/llmcompressor/pipelines/sequential/pipeline.py)

You will learn
1. Why tracing is required when compressing with recipes involving the
[Sequential Pipeline](/src/llmcompressor/pipelines/sequential/pipeline.py) and modifiers
like the [GPTQModifier](/src/llmcompressor/modifiers/quantization/gptq/base.py)
2. How to determine if your model is traceable for your dataset
3. How to modify your model definition to be traceable


## Why is Tracing Required? ##

## Determining Traceability ##
In order to determine if a model is traceable for a given dataset, you can use the
`attempt_trace` function. This function determines whether a model is traceable for a
given dataset, sequential targets list, and ignore list.


For example this script demonstrates that the `Qwen2-VL` model is traceable when
using inputs from a text-only dataset
```python
from transformers import Qwen2VLForConditionalGeneration
from llmcompressor.transformers.tracing.debug import attempt_trace

attempt_trace(
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    model_class=Qwen2VLForConditionalGeneration,
    multimodal_data=False,
    sequential_targets=["Qwen2VLDecoderLayer"],
    ignore=["lm_head", "re:visual.*"],
)
```
```
Successfully traced model into 29 subgraphs!
```

However, attempting to trace the `Qwen2-VL` with multimodal inputs (text and images)
results in a `TraceError` due to untraceable operations within the `Qwen2-VL` model
definition
```python
from transformers import Qwen2VLForConditionalGeneration
from llmcompressor.transformers.tracing.debug import attempt_trace

attempt_trace(
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    model_class=Qwen2VLForConditionalGeneration,
    multimodal_data=True,
    sequential_targets=["Qwen2VLDecoderLayer"],
    ignore=["lm_head", "re:visual.*"],
)
```
```
torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
```

### Choosing Sequential Targets ###
Sequential targets are the modules which determine the granularity of error propagation
and activation offloading when performing forward passes of the model. These are
typically the "transformer blocks" of the model, also referred to as layers with
llm-compressor.

Choosing sequential targets with a higher granularity (for example `"Linear"` instead of
`"LlamaDecoderLayer"`) will result in fewer hessians being allocated at the same time,
decreasing the memory requirements for compression. This may also increase the recovered
accuracy of the model, as compression error is propagated at a higher granularity.
However, using higher granularity sequential targets may also increase compression time,
as more time is spent offloading and onloading activations.

<p align="center">
    <img alt="Sequential Targets" src="assets/sequential_targets.jpg" height="20%" />
</p>

### Choosing Modules to Ignore ###
If your model is not traceable for your desired dataset, first consider adding any
problematic modules to the `ignore` list. Doing this prevents the model tracer from
tracing the internals of those modules, thereby avoid the untraceable operations.

For example, in this model graph, the internals of the `MllamaVisionModel` are not
traced (we don't see the individual `MllamaVisionEncoder` layers, ect.). However, we can
no longer target the modules within the `MllamaVisionModel` such as the
`MllamaVisionEncoder` as sequential targets, if any modules within the
`MllamaVisionModel` are being compressed, their hessians be all be allocated at the same
time, increasing peak memory usage.

<p align="center">
    <img alt="Ignored Modules" src="assets/ignored_modules.jpg" height="20%" />
</p>

Note that in the image above, the `multi_modal_projector` is also ignored.

## Defining your own Traceable Model Definitions ##


### Conditional Execution and Asserts ###

### Wrapping functions ###
skips problematic or shape-dependent operations, but you will not know the shape of the output

For now, must copy the original function and declare it at the module level
https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py#L1246-L1247

### Ensuring Consistent Data Types ###

### Correcting Shape Inference ###