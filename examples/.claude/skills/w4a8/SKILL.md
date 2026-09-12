---
name: w4a8
description: >
  Generate a working W4A8 quantization example script (4-bit weights + 8-bit activations,
  W4AFP8 or W4A8) and save a compressed-tensors checkpoint.
  Triggers on: "w4a8", "W4A8", "W4AFP8", "w4afp8", "int4 fp8", "int4 int8", "4-bit weight 8-bit activation".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *)]
---

# Write W4A8 Example

Generate a working Python example script that quantizes a model to a 4-bit-weight
/ 8-bit-activation scheme and saves a compressed-tensors checkpoint. This pairs
aggressive weight compression with 8-bit activation throughput.

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
2. **Scheme** — choose one (both use **dynamic** per-token activations):
   - `W4AFP8` — int4 weights (group 128) + **FP8** dynamic activations. Best on
     FP8-capable hardware (Ada/Hopper/Blackwell).
   - `W4A8` — int4 weights + **INT8** dynamic activations. For INT8 throughput
     paths (Ampere+).
3. **Algorithm** — choose one (this is what decides calibration, *not* the scheme):
   - `GPTQModifier` — learned weight rounding. **Requires a calibration dataset.**
     Best accuracy at 4-bit; the W4AFP8 example uses this.
   - `QuantizationModifier` (RTN) — round-to-nearest weights. **Data-free** (the
     activations are dynamic, so no dataset is needed); the GPT-OSS W4A8 example
     uses this.
4. **Model type** — dense, MoE, or multimodal.

## Step 2 — Choose the template

### W4AFP8 with GPTQ (calibration required)

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

# int4 weights (group 128) + FP8 dynamic per-token activations.
recipe = GPTQModifier(targets="Linear", scheme="W4AFP8", ignore=["lm_head"])

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

# Use quantization_format="pack-quantized" for vLLM compatibility.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4AFP8"
model.save_pretrained(
    SAVE_DIR, save_compressed=True, quantization_format="pack-quantized"
)
tokenizer.save_pretrained(SAVE_DIR)
```

### W4A8 with QuantizationModifier (data-free, RTN)

Because the activations are dynamic, this path needs **no calibration dataset**.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "<MODEL_ID>"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# int4 weights + INT8 dynamic per-token activations. No dataset needed.
recipe = QuantizationModifier(targets="Linear", scheme="W4A8", ignore=["lm_head"])

oneshot(model=model, recipe=recipe)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A8"
model.save_pretrained(
    SAVE_DIR, save_compressed=True, quantization_format="pack-quantized"
)
tokenizer.save_pretrained(SAVE_DIR)
```

For a channelwise-weight / asymmetric-activation variant and a large-MoE load
path, see `examples/quantization_w4a8/gpt_oss_20b_example.py`, which builds the
scheme explicitly with `config_groups` instead of the preset string. To improve
4-bit accuracy, swap `QuantizationModifier` for `GPTQModifier` (and add the
calibration dataset from the W4AFP8 template).

## Step 3 — Apply model-type adjustments

### Dense models
`ignore=["lm_head"]` is sufficient.

### MoE models
Add gate/router layers to `ignore`:
- Qwen MoE: `"re:.*mlp\.gate$"`, `"re:.*shared_expert_gate.*"`
- General: `"re:.*mlp\.router.*"`

For very large MoE models, see
`examples/quantization_w4a8/gpt_oss_20b_example.py` (the GPT-OSS model), which
also wraps loading in `load_context()`.

### Multimodal (vision / audio)
- Use `AutoProcessor` instead of `AutoTokenizer`
- Add the vision/audio towers to `ignore`
  (`"re:.*vision_tower.*"`, `"re:.*audio_tower.*"`)

## Step 4 — Write the file

Place the file under `examples/quantization_w4a8_fp8/` (W4AFP8) or
`examples/quantization_w4a8/` (W4A8).

Name the file `{model_name_slug}_example.py` (e.g. `llama3_example.py`).

Run `make style` after writing the file.

## Notes

- Both schemes use **dynamic** activations, so calibration is driven by the
  **algorithm**: GPTQ needs a dataset, RTN (`QuantizationModifier`) is data-free.
- `W4AFP8` keeps activations in FP8 (best on FP8-capable GPUs); `W4A8` uses INT8
  activations.
- Save with `quantization_format="pack-quantized"` so vLLM loads the packed int4
  weights.
- GPTQ trades calibration time for better 4-bit weight accuracy; start with RTN
  if you want a fast, data-free baseline.
