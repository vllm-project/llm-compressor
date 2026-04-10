## Qwen3.5 FP8 Example

This example quantizes the Qwen3.5-122B-A10B sparse MoE model to FP8 (weights and activations quantized to FP8) using data-free PTQ.

NOTE: This example requires `transformers >= v5`.

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Load model
2. Configure quantization algorithm and scheme
3. Apply quantization
4. Save to disk in compressed-tensors format

### 1. Load Model

```python
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"

model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

### 2. Configure Quantization Algorithm and Scheme

In this case, we are doing the following:
- Quantize the weights to FP8 with channel-wise quantization
- Quantize the activations to FP8 with dynamic per-token quantization
- Skip `lm_head`, MoE gate projections, embedding layers, shared expert gates, and linear attention layers
- MTP layers are not loaded through `Qwen3_5MoeForConditionalGeneration`, so there is no need to include them in the ignore list

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:.*lm_head",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",
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

# MTP layers are excluded from the model through Qwen3_5MoeForConditionalGeneration.
# Save them as-is from the original checkpoint into the quantized output.
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
```
