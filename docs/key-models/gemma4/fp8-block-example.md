## Gemma 4 FP8 Block Example

This example quantizes the `google/gemma-4-31B-it` model to FP8 block format using the `model_free_ptq` entrypoint. Because FP8 block quantization does not require a calibration dataset, no calibration data is needed.

The full example script can be found [here](../../../examples/model_free_ptq/gemma4_fp8_block.py).

### Code Walkthrough

### 1. Quantize Model to FP8 Block Format

```python
from llmcompressor import model_free_ptq

MODEL_ID = "google/gemma-4-31B-it"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8_BLOCK"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=["re:.*vision.*", "lm_head", "re:.*embed_tokens.*"],
    max_workers=8,
    device="cuda:0",
)
```

The `ignore` list skips the vision tower, `lm_head`, and embedding layers, which are kept in their original precision.