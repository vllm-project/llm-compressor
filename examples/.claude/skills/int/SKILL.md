---
name: int
description: >
  Generate a working integer quantization example script (W4A16 or W8A8) and save a compressed-tensors checkpoint.
  Triggers on: "int", "W4A16", "W8A8", "GPTQ", "SmoothQuant", "AWQ", "int8", "int4", "integer quantization".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *)]
---

# Write INT Example

Generate a working Python example script that quantizes a model to an integer scheme (W4A16 or W8A8) and saves a compressed-tensors checkpoint.

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
2. **Scheme** — choose one:
   - `W4A16` — weights int4 group-128, activations fp16. Weight-only quantization, good memory savings.
   - `W8A8` — weights int8 static per-channel, activations int8 dynamic per-token. Good throughput on Ampere+.
3. **Algorithm** — choose one:
   - `GPTQModifier` — learned weight rounding. Requires calibration. Works for both W4A16 and W8A8.
   - `SmoothQuantModifier` + `GPTQModifier` — SmoothQuant migrates activation outliers into weights before GPTQ. Recommended for W8A8.
   - `AWQModifier` + `QuantizationModifier` — AWQ scales weights before quantization. Requires calibration. Works for W4A16.
4. **Model type** — dense, MoE, or multimodal (vision/audio)

## Step 2 — Choose the template

### W4A16 with GPTQModifier (calibration required)

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

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

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

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

### W8A8 with GPTQModifier + SmoothQuant (calibration required, better accuracy)

Adding `SmoothQuantModifier` before GPTQ smooths activation outliers and typically improves W8A8 accuracy.

```python
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
```

Everything else (data loading, `oneshot` call, save) is identical to the W4A16 template above. Update `SAVE_DIR` suffix to `-W8A8`.

### W4A16 with AWQModifier + QuantizationModifier (calibration required)

AWQ scales weights to reduce quantization error before applying W4A16. Use `duo_scaling="both"` to scale both inputs and weights.

```python
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

MODEL_ID = "<MODEL_ID>"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

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

recipe = [
    AWQModifier(duo_scaling="both"),
    QuantizationModifier(
        targets=["Linear"],
        scheme="W4A16_ASYM",
        ignore=["lm_head"],  # extend per model type — see Step 3
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-AWQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Step 3 — Apply model-type adjustments

### Dense models
`ignore=["lm_head"]` is sufficient in most cases.

### MoE models
Add gate/router and norm layers to `ignore`. For models that need a custom loading path, wrap the load in `load_context`:

```python
from llmcompressor.utils import load_context

with load_context(SpecificModelClass):
    model = SpecificModelClass.from_pretrained(MODEL_ID)
```

Common MoE ignore additions:
- Qwen MoE: `"re:.*mlp.gate$"`, `"re:.*shared_expert_gate.*"`, `"re:.*norm.*"`, `"re:.*embed_tokens.*"`
- General: `"re:.*mlp.router.*"`, `"re:.*norm.*"`

### Multimodal (vision / audio)
- Use `AutoProcessor` instead of `AutoTokenizer`
- Use the model-specific class and `load_context` if needed
- Add to `ignore`:
  - Vision: `"re:.*vision_tower.*"`, `"re:.*vision_model.*"`, `"re:.*multi_modal_projector.*"`
  - Audio: `"re:.*audio_tower.*"`

## Step 4 — Write the file

Place the file under `examples/`:
- `quantization_w4a16/` for W4A16 (GPTQ or AWQ)
- `quantization_w8a8_int8/` for W8A8 (GPTQ or SmoothQuant+GPTQ)
- `awq/` when the primary algorithm is AWQ

Name the file `{model_name_slug}_example.py` (e.g. `llama3_example.py`).

Run `make style` after writing the file.

## Notes
- W4A16: requires calibration; use GPTQ for simplicity or AWQ for better weight scaling.
- W8A8: SmoothQuant+GPTQ is the recommended path — SmoothQuant migrates outliers into weights, making activation quantization significantly more accurate.
- AWQ uses `W4A16_ASYM` (asymmetric) by default; swap to `W4A16` for symmetric if needed.
- For W4A16, add `quantization_format="pack-quantized"` to `save_pretrained` if vLLM requires packed int4 format.
