## Kimi-K2.6 NVFP4 Example

### Code Walkthrough

The original [Kimi K2.6 checkpoint](https://huggingface.co/moonshotai/Kimi-K2.6) ships in a quantized format with 4-bit integer weights. 
In order to create an NVFP4 checkpoint that can leverage NVIDIA's 4-bit floating point kernels, we must first dequantize to full-precision (bfloat16), then quantize to the desired NVFP4 format. Note that this requires saving the full-precision model to an intermediate directory.
Let's walk through the main steps of the quantization process:
1. Dequantize model
2. Apply quantization to full-precision checkpoint

The full example script can be found in the examples [here](../../../examples/disk_offloading/kimi_k26_example.py).

### 1. Dequantize Model

```python
from compressed_tensors.entrypoints.convert import (
    CompressedTensorsDequantizer,
    convert_checkpoint,
)

MODEL_ID = "moonshotai/Kimi-K2.6"
DEQUANTIZED_SAVE_DIR = "Kimi-K2.6-bf16"

ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*self_attn.*",
    "re:.*embed_tokens$",
    # ignore anything not in language_model
    "re:.*mm_projector.*",
    "re:.*vision.*",
]

# Convert to dense bfloat16 format
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=DEQUANTIZED_SAVE_DIR,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        ignore=ignore,
    ),
    max_workers=4,
)
```

### 2. Apply Quantization

Once dequantized, the model can be quantized to NVFP4 via oneshot. NVFP4 uses static activation quantization, so a calibration dataset is required for oneshot.
Because the model is one trillion parameters, we leverage the `compressed_tensors.offload` module with disk offloading to run the calibration dataset
through the model. The snippet below was run successfully on a single H100x80GB GPU and 500GB CPU RAM.

```python
from compressed_tensors.offload import load_offloaded_model
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SAVE_DIR = "Kimi-K2.6-NVFP4"

# Quantize bfloat16 checkpoint to NVFP4, limiting CPU RAM usage to 500GB
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        DEQUANTIZED_SAVE_DIR,
        dtype="auto",
        device_map="auto_offload",
        max_memory={"cpu": 500e9},
        trust_remote_code=True,
        offload_folder="./offload_folder",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        DEQUANTIZED_SAVE_DIR, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        DEQUANTIZED_SAVE_DIR, trust_remote_code=True
    )

# Select calibration dataset.
DATASET_ID = "ultrachat-200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

# Configure the quantization algorithm to run.
#   * quantize the weights to NVFP4
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=ignore,
)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    dataset=DATASET_ID,
    splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

The dequantized model can be deleted once step 2 completes.