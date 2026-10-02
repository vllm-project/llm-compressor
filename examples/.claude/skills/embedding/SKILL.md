---
name: embedding
description: >
  Generate a working embedding quantization example script (weight-only int4/int8 of the
  embedding table) and save a compressed-tensors checkpoint.
  Triggers on: "embedding quantization", "quantize embedding", "embed_tokens", "embedding table", "quantize embeddings".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *)]
---

# Write Embedding Example

Generate a working Python example script that quantizes a model's input embedding
table (weight-only) and saves a compressed-tensors checkpoint.

Embedding quantization shrinks the lookup table, which is a large share of
parameters in small/medium models. It is weight-only (a lookup has no
activations), so it is **data-free** — no calibration dataset is required.

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
2. **Bit width / group size** — e.g. int4 group 64 (default), or int8.
3. **Pair with Linear quantization?** — embeddings are often quantized alongside
   a W4A16 / W8A8 scheme on the `Linear` layers (add a second config group).
4. **Model type** — dense, MoE, or multimodal.

## Step 2 — Choose the template

The target is the **`Embedding` module class** (not a name regex), which keeps the
recipe portable across architectures regardless of the module's name
(`model.embed_tokens`, `gpt_neox.embed_in`, ...).

```python
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "<MODEL_ID>"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Quantize the input embedding table to int4, group size 64. Weight-only, so this
# is data-free.
recipe = QuantizationModifier(
    config_groups={
        "embedding": {
            "targets": ["Embedding"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 64,
            },
        }
    }
)

oneshot(model=model, recipe=recipe)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-embedding-W4A16-G64"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Step 3 — Apply model-type adjustments

### Pair with Linear-layer quantization
Add a second config group targeting `Linear` to quantize weights too, e.g.:

```python
recipe = QuantizationModifier(
    config_groups={
        "embedding": {
            "targets": ["Embedding"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True,
                        "strategy": "group", "group_size": 64},
        },
        "linear": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True,
                        "strategy": "group", "group_size": 128},
        },
    },
    ignore=["lm_head"],
)
```

### MoE / multimodal
Embedding quantization targets the `Embedding` class directly, so MoE routers and
vision/audio towers are unaffected. Use `AutoProcessor` for multimodal models.

## Step 4 — Write the file

Place the file under `examples/quantization_embedding/`.

Name the file `{model_name_slug}_example.py` (e.g. `llama3_example.py`).

Run `make style` after writing the file.

## Notes

- Embedding quantization is **data-free** — `oneshot(model=model, recipe=recipe)`
  with no `dataset` is correct.
- Target the `Embedding` **class**, not a module-name regex, for portability.
- Most impactful on models where the embedding/`lm_head` is a large fraction of
  parameters (small models, large vocabularies).
