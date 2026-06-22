## Gemma 4 NVFP4 Example

This example quantizes the `google/gemma-4-31B-it` multimodal model to NVFP4 (weights and activations quantized to FP4) using data-driven PTQ with calibration. The vision and audio layers are skipped during quantization.

The full example script can be found [here](../../../examples/multimodal_vision/gemma4_example.py).

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Load model
2. Configure quantization algorithm and scheme
3. Prepare calibration dataset
4. Apply quantization
5. Confirm generations of the quantized model look sane
6. Save to disk in compressed-tensors format

### 1. Load Model

Load the model and processor using `Gemma4ForConditionalGeneration`:

```python
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = "google/gemma-4-31B-it"
model = Gemma4ForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
```

### 2. Configure Quantization Algorithm and Scheme

We quantize weights and activations to FP4, skipping vision, audio, embedding, and `lm_head` layers:

```python
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=["re:.*vision.*", "re:.*audio.*", "lm_head", "re:.*embed.*"],
    ),
]
```

### 3. Prepare Calibration Dataset

We use the `mit-han-lab/pile-val-backup` dataset for calibration with 32 samples and a maximum sequence length of 2048 tokens:

```python
from datasets import load_dataset

BATCH_SIZE = 1
NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 2048

DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"


def get_calib_dataset(processor):
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*10}]",
    )

    def preprocess(example):
        return {
            "input_ids": processor.tokenizer.encode(example["text"].strip())[
                :MAX_SEQUENCE_LENGTH
            ]
        }

    ds = (
        ds.shuffle(seed=42)
        .map(preprocess, remove_columns=ds.column_names)
        .filter(lambda example: len(example["input_ids"]) >= MAX_SEQUENCE_LENGTH)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )
    return ds
```

### 4. Apply Quantization

```python
oneshot(
    model=model,
    processor=processor,
    dataset=get_calib_dataset(processor),
    recipe=recipe,
    batch_size=BATCH_SIZE,
    shuffle_calibration_samples=False,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
```

### 5. Confirm Generations of the Quantized Model Look Sane

```python
import requests
from compressed_tensors.offload import dispatch_model
from PIL import Image

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe the animal in this image\n"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image_url = "http://images.cocodataset.org/train2017/000000231895.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100, disable_compile=True)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")
```

### 6. Save to Disk in Compressed-Tensors Format

```python
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
```
