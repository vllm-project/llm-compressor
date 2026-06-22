---
name: mxfp4
description: >
  Generate a working MXFP4 quantization example script targeting AMD MI300X and save a compressed-tensors checkpoint.
  Triggers on: "mxfp4", "MXFP4", "mx fp4", "fp4 amd", "mi300x quantization".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *)]
---

# Write MXFP4 Example

Generate a working Python example script that quantizes a model to MXFP4 (weights and activations fp4 with MX block scaling) and saves a compressed-tensors checkpoint.

Hardware requirement: AMD MI300X for inference. Quantization can run on any GPU.

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
2. **Algorithm** — choose one:
   - `QuantizationModifier` — simple PTQ, no calibration data required.
   - `GPTQModifier` — learned weight rounding, requires calibration data. Better accuracy.
3. **Model type** — dense, MoE, or multimodal (vision/audio)

## Step 2 — Choose the template

### QuantizationModifier (no calibration)

```python
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "<MODEL_ID>"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="MXFP4",
    ignore=["lm_head"],  # extend per model type — see Step 3
)

oneshot(model=model, recipe=recipe)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

### GPTQModifier (with calibration)

```python
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID = "<MODEL_ID>"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(preprocess)
ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = GPTQModifier(
    targets="Linear",
    scheme="MXFP4",
    ignore=["lm_head"],  # extend per model type — see Step 3
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4-GPTQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Step 3 — Apply model-type adjustments

### Dense models
`ignore=["lm_head"]` is sufficient in most cases.

### MoE models
Add gate/router layers to `ignore`. For models that need a custom loading path, wrap the load in `load_context`:

```python
from llmcompressor.utils import load_context

with load_context(SpecificModelClass):
    model = SpecificModelClass.from_pretrained(MODEL_ID)
```

Common MoE ignore additions:
- Qwen MoE: `"re:.*mlp.gate$"`, `"re:.*shared_expert_gate.*"`
- General: `"re:.*mlp.router.*"`

### Multimodal (vision / audio)
- Use `AutoProcessor` instead of `AutoTokenizer`
- Use the model-specific class and `load_context` if needed
- Add to `ignore`:
  - Vision: `"re:.*vision_tower.*"`, `"re:.*vision_model.*"`, `"re:.*multi_modal_projector.*"`
  - Audio: `"re:.*audio_tower.*"`

## Step 4 — Write the file

Place the file under `examples/quantization_w4a4_mxfp4/`.

Name the file `{model_name_slug}_example.py` (e.g. `llama3_mxfp4.py`).

Run `make style` after writing the file.

## Notes
- With `QuantizationModifier`: `oneshot(model=model, recipe=recipe)` with no dataset is correct — no calibration data is needed.
- With `GPTQModifier`: 512 samples at 2048 seq-len is a good default.
- Both target AMD MI300X for inference.
