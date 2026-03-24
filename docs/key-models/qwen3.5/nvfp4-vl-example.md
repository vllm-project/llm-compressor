## Qwen3.5 NVFP4A16 Vision-Language Example

This example quantizes the Qwen3.5-27B vision-language model to NVFP4A16 (weights quantized to FP4 with per-group-16 granularity, activations in FP16) using data-free PTQ.

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Load model
2. Configure quantization algorithm and scheme
3. Apply quantization
4. Run sample generation
5. Save to disk in compressed-tensors format

### 1. Load Model

```python
from compressed_tensors.offload import dispatch_model
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model.
MODEL_ID = "Qwen/Qwen3.5-27B"
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
```

### 2. Configure Quantization Algorithm and Scheme

In this case, we are doing the following:
- Quantize the weights to FP4 with per-group-16 granularity via data-free PTQ
- Skip the visual encoder, `lm_head`, and linear attention layers (Gated DeltaNet fused projections are incompatible with NVFP4)
- MTP layers are not loaded through `Qwen3_5ForConditionalGeneration`, so there is no need to include them in the ignore list

```python
# No need to include mtp layers as they are not loaded
# through Qwen3_5ForConditionalGeneration
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4A16",
    ignore=[
        "lm_head",
        "re:.*visual.*",
        "re:.*linear_attn.*",
    ],
)
```

### 3. Apply Quantization

```python
oneshot(model=model, recipe=recipe)
```

### 4. Run Sample Generation

```python
print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
messages = [{"role": "user", "content": "Hello my name is"}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================\n\n")
```

### 5. Save to Disk in Compressed-Tensors Format

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

# MTP layers are excluded from the model through Qwen3_5ForConditionalGeneration
# Save them as-is from the original checkpoint into the quantized output.
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
```
