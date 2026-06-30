---
name: nvfp4
description: >
  Generate a working NVFP4 quantization example script targeting H100/Blackwell and save a compressed-tensors checkpoint.
  Triggers on: "nvfp4", "NVFP4", "nv fp4", "fp4 nvidia", "fp4 hopper", "fp4 blackwell", "h100 fp4".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *)]
---

# Write NVFP4 Example

Generate a working Python example script that quantizes a model to NVFP4 (weights fp4 per-group-16, activations fp4 with global scale) and saves a compressed-tensors checkpoint.

Hardware requirement: H100 / Blackwell (sm90+) for inference. Quantization can run on any GPU.

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`)
2. **Algorithm** — choose one:
   - `QuantizationModifier` — simple PTQ, no weight optimization. Faster.
   - `GPTQModifier` — learned weight rounding via GPTQ. Better accuracy, slower.
3. **Model type** — dense, MoE, or multimodal (vision/audio)

NVFP4 always requires calibration data.

## Step 2 — Choose the template

### QuantizationModifier (PTQ)

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "<MODEL_ID>"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

NUM_CALIBRATION_SAMPLES = 256
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

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head", "re:.*embed_tokens$"],  # extend per model type — see Step 3
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

### GPTQModifier

Replace only the recipe line:

```python
from llmcompressor.modifiers.gptq import GPTQModifier

recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head", "re:.*embed_tokens$"],  # extend per model type — see Step 3
)
```

Everything else (data loading, `oneshot` call, save) is identical to the `QuantizationModifier` template.

## Step 3 — Apply model-type adjustments

### Dense models
`ignore=["lm_head", "re:.*embed_tokens$"]`

### MoE models
Add gate/router layers to `ignore`. For models that need a custom loading path, wrap the load in `load_context`:

```python
from llmcompressor.utils import load_context

with load_context(SpecificModelClass):
    model = SpecificModelClass.from_pretrained(MODEL_ID)
```

Common MoE ignore additions:
- Qwen MoE: `"re:.*mlp.gate$"`, `"re:.*shared_expert_gate.*"`, `"re:.*linear_attn.*"`
- Llama4: `"re:.*router"`, `"re:.*self_attn"`, `"Llama4TextAttention"`
- General: `"re:.*mlp.router.*"`, `"re:.*self_attn.*"`

For large MoE models, calibrate one expert block at a time:
```python
oneshot(..., sequential_targets=["Llama4TextMLP"])
```

To calibrate all experts (not just sampled routed ones):
```python
oneshot(..., moe_calibrate_all_experts=True)
```

### Multimodal (vision / audio)
- Use `AutoProcessor` instead of `AutoTokenizer`
- Use the model-specific class and `load_context` if needed
- Add to `ignore`:
  - Vision: `"re:.*vision_tower.*"`, `"re:.*vision_model.*"`, `"re:.*multi_modal_projector.*"`
  - Audio: `"re:.*audio_tower.*"`
  - Embedding projections: `"re:.*embed_vision.*"`, `"re:.*embed_audio.*"`
- Preprocessing must use `processor.apply_chat_template` with `return_dict=True`; pass a `data_collator` to `oneshot`

## Step 4 — Write the file

Place the file under `examples/quantization_w4a4_fp4/`.

Name the file `{model_name_slug}_example.py` (e.g. `llama3_example.py`, `qwen3_5_example.py`).

Run `make style` after writing the file.

## Notes
- 256 calibration samples at 2048 sequence length is a good default; increase samples if accuracy drops.
- MTP (multi-token prediction) layers in some Qwen models are excluded from the standard model class. Save them separately after quantization using `save_mtp_tensors_to_checkpoint` from `compressed_tensors`.
- `save_compressed=True` is not required — NVFP4 checkpoints save correctly without it, but it can be added.
