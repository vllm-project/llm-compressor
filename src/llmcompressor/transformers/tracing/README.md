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
Before defining your own traceable model definition, make sure that the untraceable
parts of your model are not a part of a module that can be
[ignored](#choosing-modules-to-ignore).

To define your own traceable model definition, follow the steps below:
1. Copy the original model definition into the [tracing folder](/src/llmcompressor/transformers/tracing/).
The original model definition can usually be found in the [`transformers/models`](https://github.com/huggingface/transformers/tree/main/src/transformers/models)
folder or the `modeling_X.py` file when using models with remote code.
2. Add your new model class to [tracing/\_\_init\_\_.py](/src/llmcompressor/transformers/tracing/__init__.py).
3. Use the `attempt_trace` function as show in [Determining Traceability](#determining-traceability)
to find the untraceable line of code in your model. **Remember to replace the original
`model_class` with your own imported custom traceable model definition**.
4. Find the untraceable line of code in your model definition and modify the code to
make it traceable. Examples of how to do this for each of the common errors can be found
below. If you encounter a tracing issue which is not documented below, please create an
issue!
5. Add a comment above the line which has been modified explaining why the operation is
untraceable. For example, `# TRACING: Assume that the attention mask is always present`
6. Repeat steps 3-5 until all of the untraceable operations have been replaced with
traceable operations.
7. Once your model traces successfully, remove any class definitions you did not use and
import them if necessary. Note that this cannot be done for models with remote code.
8. Commit your changes to a branch and open a PR so that others like yourself can
benefit from the changes! LLM Compressor is an open-source project that relies on
community contribution to support the wide range of model architectures available on
huggingface. P.S., remember to add `# vllm-project: no copyright` underneath any
copyright notices at the top of the file. TODO: revisit the tone here

### Conditional Logic and Asserts ###
```
torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
```

```python3
if n_image_tokens != n_image_features:
    raise ValueError(
        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    )
```

### Conditional Iteration ###
```
torch.fx.proxy.TraceError: Proxy object cannot be iterated. This can be attempted when the Proxy is used in a loop or as a *args or **kwargs function argument. See the torch.fx docs on pytorch.org for a more detailed explanation of what types of control flow can be traced, and check out the Proxy docstring for help troubleshooting Proxy iteration errors
```

https://pytorch.org/docs/main/fx.html#torch.fx.Proxy


### Wrapping functions ###
When tracing the [`MllamaForConditionalGeneration`](/src/llmcompressor/transformers/tracing/mllama.py)
architecture, we encounter a `TraceError` on this line:
```python3
batch_size, text_total_length, *_ = cross_attention_mask.shape
# torch.fx.proxy.TraceError: Proxy object cannot be iterated
```

In this case, making this line traceable is fairly trivial
```python3
batch_size, text_total_length = cross_attention_mask.shape[:2]
```

However, 

Since this function does not output any variable whose shapes require inference
(see [Correcting Shape Inference](#correcting-shape-inference)), we can simply wrap the
function which performs the untraceable operation. This is equivalent to adding the
function to an "ignore" list which ensures that its internals are not traced within the
model graph.

```python3
@torch.fx.wrap
def _prepare_cross_attention_mask(...) -> ...:
    ...
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    ...
```

<p align="center">
    <img alt="Wrapped Function" src="assets/wrapped_function.jpg" height="20%" />
</p>

TODO: In the future, 
https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py#L1246-L1247

### Correcting Shape Inference ###
When performing tracing with LLM Compressor, the shapes of some variables are assumed
based on the shapes provided by a sample from the dataset. This is done to ensure some
models which include basic [Conditional Logic and Asserts](#conditional-logic-and-asserts)
traceable without major changes to the model definition.

However, there are some instances where the shape inferences is not properly
implemented, leading to some variables whose shape is unknown. This is not always a
problem, unless those variables are used for conditional logic later during execution.
In these cases, rather than fixing every instance of condition logic, we can inject our
own knowledge of variable shapes.

```python3
inputs_embeds_masked = inputs_embeds.masked_scatter(special_image_mask, image_features)
# TRACING: install metadata
inputs_embeds_masked = maybe_install_metadata_inputs_embeds(inputs_embeds_masked, inputs_embeds, special_image_mask, image_features)
```

### Ensuring Consistent Data Types ###

```python3
legacy_processing = False
legacy_processing = (
    (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
) or (input_ids.shape[-1] == 1 and pixel_values is not None)
```