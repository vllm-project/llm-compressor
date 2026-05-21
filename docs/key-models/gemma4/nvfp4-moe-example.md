## Gemma 4 NVFP4 MoE Example

This example quantizes the `google/gemma-4-26B-A4B-it` sparse MoE model to NVFP4 (weights and activations quantized to FP4) using data-driven PTQ with calibration. The `SequentialGemma4TextExperts` modules are applied automatically by the pipeline to enable proper expert handling and vLLM compatibility.

The full example script can be found [here](../../../examples/quantization_w4a4_fp4/gemma4_example.py).

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
from datasets import load_dataset
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "google/gemma-4-26B-A4B-it"

model = Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

### 2. Configure Quantization Algorithm and Scheme

We quantize weights and activations to FP4, skipping the `lm_head`, embedding layers, MoE router, and vision tower:

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        "re:.*embed.*",
        "re:.*router",
        "re:.*vision_tower.*",
    ],
)
```

### 3. Prepare Calibration Dataset

We use the `neuralmagic/calibration` dataset with 20 samples and a maximum sequence length of 8192 tokens:

```python
DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 8192

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess_function(example):
    messages = []
    for message in example["messages"]:
        messages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )
    return processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }
```

### 4. Apply Quantization

MoE calibration is handled automatically by the pipeline via `SequentialGemma4TextExperts`:

```python
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)
```

### 5. Save to Disk in Compressed-Tensors Format

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
```
