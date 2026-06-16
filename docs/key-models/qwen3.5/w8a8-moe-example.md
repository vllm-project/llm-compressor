## Qwen3.5 W8A8 MoE Example

This example quantizes Qwen3.5/3.6 sparse MoE models (e.g. `Qwen/Qwen3.5-35B-A3B` or `Qwen/Qwen3.6-35B-A3B`) to W8A8 (INT8 weights with static per-channel scales and INT8 activations with dynamic per-token scales) using SmoothQuant + GPTQ with calibration.

> **Note:** Qwen3.6 shares the same model architecture as Qwen3.5. Use the same script and swap `MODEL_ID`. Please ensure transformers v5 is installed.

### Code Walkthrough

1. Load model
2. Prepare calibration dataset
3. Configure SmoothQuant + GPTQ W8A8
4. Apply quantization
5. Save to disk in compressed-tensors format

### 1. Load Model

```python
import torch
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"  # or "Qwen/Qwen3.6-35B-A3B"

model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

### 2. Prepare Calibration Dataset

We use the UltraChat dataset for calibration with 256 samples and a maximum sequence length of 4096 tokens.

```python
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 4096

ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]",
)
ds = ds.select_columns(["messages"])
ds = ds.shuffle(seed=42)
```

### 3. Configure SmoothQuant + GPTQ W8A8

- Apply SmoothQuant to make activations easier to quantize
- Quantize weights to INT8 with GPTQ (static per channel)
- Quantize activations to INT8 (dynamic per token)
- Skip the `lm_head`, visual encoder, MoE router gate, embedding tokens, shared expert gate, and linear attention layers (Gated DeltaNet; same as NVFP4 MoE examples)

```python
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=[
            "re:.*lm_head",
            "re:visual.*",
            "re:model.visual.*",
            "re:.*mlp.gate$",
            "re:.*embed_tokens$",
            "re:.*shared_expert_gate$",
            "re:.*linear_attn.*",
        ],
    ),
]
```

### 4. Apply Quantization

For MoE models, use `moe_calibrate_all_experts=True` so every expert sees calibration data.

```python
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
    data_collator=data_collator,
)
```

### 5. Save to Disk

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
```

### Data-Free Alternative

For a data-free RTN W8A8 pass (no calibration dataset), use the model-free PTQ entrypoint:

```bash
python examples/quantization_w8a8_int8/qwen3_6_example.py --algorithm rtn
```

Swap `MODEL_ID` in that script for Qwen3.6.

### Full Example Script

See [examples/quantization_w8a8_int8/qwen3_6_example.py](../../../examples/quantization_w8a8_int8/qwen3_6_example.py).
