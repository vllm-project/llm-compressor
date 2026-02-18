## Kimi-K2 FP8 Example

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Load model
2. Apply quantization

### 1. Load Model

```python
from llmcompressor import model_free_ptq

MODEL_ID = "unsloth/Kimi-K2-Thinking-BF16"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"
```

### 2. Apply Quantization

Once quantized, the model is saved. This uses compressed-tensors to the SAVE_DIR.

```python
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*gate$",
        "lm_head",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "model.embed_tokens",
    ],
    max_workers=15,
    device="cuda:0",
)
```