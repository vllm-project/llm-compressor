# API reference

!!! info
    This is the auto-generated API documentation for LLM Compressor.
    Browse the `llmcompressor` module tree to explore all available classes, functions, and configuration options.

## Using the API

The primary entry point for most users is the `oneshot` function:

```python
from llmcompressor.transformers import oneshot

oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
)
```

For advanced usage, you can configure individual modifiers and apply them directly to models.
See the [Examples](https://github.com/vllm-project/llm-compressor/tree/main/examples) section for detailed usage patterns.
