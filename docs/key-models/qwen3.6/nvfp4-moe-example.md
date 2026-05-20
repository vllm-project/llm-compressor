## Qwen3.6 NVFP4 MoE Example

This example quantizes the Qwen3.6-35B-A3B sparse MoE model to NVFP4 (weights and activations quantized to FP4) using data-driven PTQ with calibration.

> **Note:** Qwen3.6 shares the same model architecture as Qwen3.5, so the quantization process is identical. Please ensure transformers v5 is installed.

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Load model
2. Configure quantization algorithm and scheme
3. Prepare calibration dataset
4. Apply quantization
5. Save to disk in compressed-tensors format

### 1. Load Model

```python
import torch
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE: This example requires transformers >= v5

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

# Load model.
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

### 2. Configure Quantization Algorithm and Scheme

In this case, we are doing the following:
- Quantize the weights and activations to FP4 via data-driven PTQ
- Skip the `lm_head`, visual encoder, MLP gate, embedding tokens, shared expert gate, and linear attention layers (Gated DeltaNet fused projections are incompatible with NVFP4)
- MTP layers are not loaded through `Qwen3_5MoeForConditionalGeneration`, so there is no need to include them in the ignore list

```python
# No need to include mtp layers as they are not loaded
# through Qwen3_5MoeForConditionalGeneration
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)
```

### 3. Prepare Calibration Dataset

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


def preprocess_function(example):
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in example["messages"]
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        processor_kwargs={
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": MAX_SEQUENCE_LENGTH,
            "add_special_tokens": False,
        },
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}
```

### 4. Apply Quantization

For MoE models, we use `moe_calibrate_all_experts=True` to ensure all experts are calibrated during quantization.

```python
# Apply quantization.
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

### 5. Save to Disk in Compressed-Tensors Format

```python
# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

# MTP layers are excluded from the model through Qwen3_5MoeForConditionalGeneration
# Save them as-is from the original checkpoint into the quantized output.
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
```
