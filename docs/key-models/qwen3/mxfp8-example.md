## Qwen3 MXFP8 Examples

LLM Compressor supports MXFP8 quantization for Qwen3 models in two modes:

- **W8A8 (MXFP8)**: Both weights and activations are quantized to MXFP8
- **W8A16 (MXFP8A16)**: Only weights are quantized to MXFP8; activations remain in 16-bit

> **Note:** MXFP8 support is experimental.

---

### W8A8: MXFP8 Weights and Activations

Quantizes both weights and activations to MXFP8 via PTQ.

**Script:** [`experimental/mxfp8/qwen3_example_w8a8_mxfp8.py`](../../../experimental/mxfp8/qwen3_example_w8a8_mxfp8.py)

#### Code Walkthrough

##### 1. Load Model

```python
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

##### 2. Configure Quantization

In this case, we quantize the weights and activations to MXFP8 via PTQ:

```python
recipe = QuantizationModifier(
    targets="Linear", scheme="MXFP8", ignore=["lm_head"]
)
```

##### 3. Apply Quantization and Save

```python
oneshot(model=model, recipe=recipe)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP8"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

---

### W8A16: MXFP8 Weights Only

Quantizes only weights to MXFP8 while retaining activations in 16-bit precision.

**Script:** [`experimental/mxfp8/qwen3_example_w8a16_mxfp8.py`](../../../experimental/mxfp8/qwen3_example_w8a16_mxfp8.py)

#### Code Walkthrough

##### 1. Load Model

```python
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

##### 2. Configure Quantization

In this case, we quantize the weights to MXFP8 via PTQ (activations remain in 16-bit):

```python
recipe = QuantizationModifier(
    targets="Linear", scheme="MXFP8A16", ignore=["lm_head"]
)
```

##### 3. Apply Quantization and Save

```python
oneshot(model=model, recipe=recipe)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP8A16"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```
