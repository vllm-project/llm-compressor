## Qwen3 FP8 Example

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Load model
2. Configure quantization algorithm and scheme
3. Apply quantization
4. Save to disk in compressed-tensors format

### 1. Load Model

```python
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE: Requires a minimum of transformers 4.57.0

MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

### 2. Configure Quantization Algorithm and Scheme

In this case, we are doing the following:
 * quantize the weights to fp8 with channel-wise quantization
 * quantize the activations to fp8 with dynamic token activations

NOTE: Only datafree quantization is supported for Qwen3-VL-MoE currently

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
    ],
)
```

### 3. Apply Quantization

```python
oneshot(model=model, recipe=recipe)
```

### 4. Save to Disk in Compressed-Tensors Format

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-DYNAMIC"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```