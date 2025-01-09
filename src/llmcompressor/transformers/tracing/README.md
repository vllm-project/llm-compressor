# Tracing #
LLM Compressor see the [Sequential Pipeline](/src/llmcompressor/pipelines/sequential/pipeline.py)


## Testing Traceability ##


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
Attempting trace
    model_id=Qwen/Qwen2-VL-2B-Instruct
    dataset=ultrachat-200k
    split=test_sft[:1]
    inputs=dict_keys(['input_ids', 'attention_mask'])
    sequential_targets=['Qwen2VLDecoderLayer']
    ignore=['lm_head', 're:visual.*']

Successfully traced model into 29 subgraphs!
```

However, attempting to trace 

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

## Defining your own Traceable Model Definitions ##




### ignoring
First, try modifying sequential_targets and ignore

### Removing asserts

### Changing data types

### wrapping functions
skips problematic or shape-dependent operations, but you will not know the shape of the output

For now, must copy the original function and declare it at the module level
https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py#L1246-L1247

### injecting shapes via metadata