---
name: model-free-ptq
description: >
  Generate a working model_free_ptq example script for data-free quantization of very large
  models (FP8, NVFP4A16, MXFP4A16) directly from safetensors, without a transformers model definition.
  Triggers on: "model_free_ptq", "model free ptq", "data-free ptq", "quantize huge model", "no model definition", "kimi", "deepseek large".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *)]
---

# Write model_free_ptq Example

Generate a working Python example script that quantizes a model with
`model_free_ptq` and saves a compressed-tensors checkpoint.

`model_free_ptq` works directly on the checkpoint safetensors — it does **not**
load the model through transformers. Use it when:

1. the model has **no transformers model definition** (e.g. a brand-new arch), or
2. the model is **very large** (e.g. Kimi-K2, DeepSeek) and `oneshot` runs into
   memory issues.

It supports **data-free schemes only** (no calibration): FP8 dynamic/block and the
weight-only microscale schemes (NVFP4A16 / MXFP4A16).

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace stub or local path (passed directly, not loaded
   through transformers).
2. **Scheme** — a data-free scheme: `FP8_DYNAMIC`, `FP8_BLOCK`, `NVFP4A16`, or
   `MXFP4A16`.
3. **ignore list** — modules to keep in full precision (`lm_head`,
   `model.embed_tokens`, MoE gates, and any blocks incompatible with the scheme's
   block size).
4. **device** / **max_workers** — e.g. `device="cuda:0"`, `max_workers=15`.

## Step 2 — Choose the template

### FP8 (FP8_DYNAMIC / FP8_BLOCK)

```python
from llmcompressor import model_free_ptq

MODEL_ID = "<MODEL_ID>"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",  # or FP8_DYNAMIC
    ignore=[
        "model.embed_tokens",
        "lm_head",
        # add MoE gates / block-size-incompatible layers per model — see Step 3
    ],
    max_workers=15,
    device="cuda:0",
)
```

### Microscale (NVFP4A16 / MXFP4A16) — requires a reindex step

Microscale schemes fuse global scales across fused modules (qkv, gate_up), so the
safetensors must first be reindexed to put fused modules in the same file:

```bash
llmcompressor.reindex_fused_weights \
    <MODEL_ID> <MODEL_ID_slug>-reindexed --num_workers=10
```

Then run `model_free_ptq` on the reindexed files:

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="<MODEL_ID_slug>-reindexed",
    save_directory="<MODEL_ID_slug>-NVFP4A16",
    scheme="NVFP4A16",  # or MXFP4A16
    ignore=["lm_head", "re:.*gate$", "model.embed_tokens"],
    max_workers=15,
    device="cuda:0",
)
```

## Step 3 — Apply model-type adjustments

### Dense models
`ignore=["model.embed_tokens", "lm_head"]` is usually sufficient.

### MoE models
Add the router/gate and any block-size-incompatible projections to `ignore`, e.g.
for DeepSeek/Kimi-style models:
`"re:.*gate$"`, `"re:.*kv_a_proj_with_mqa$"`, `"re:.*q_a_proj$"`. See the Kimi-K2
and DeepSeek examples under `examples/model_free_ptq/`.

### FP8_BLOCK note
Block quantization needs dimensions divisible by the block (commonly 128); ignore
layers that are not (the example ignore lists show the common ones per model).

## Step 4 — Write the file

Place the file under `examples/model_free_ptq/`.

Name the file `{model_name_slug}_{scheme_slug}.py`
(e.g. `qwen3_fp8_block.py`, `kimi_k2_thinking_nvfp4a16.py`).

Run `make style` after writing the file.

## Notes

- Data-free schemes only — `model_free_ptq` does not take a calibration dataset.
- The model stub is passed **directly** (not loaded through transformers); this is
  what lets it handle models without a transformers definition.
- For microscale schemes, the `llmcompressor.reindex_fused_weights` step is
  required before quantizing.
- This is the recommended path for the largest models (Kimi-K2, DeepSeek,
  Mistral-Large); for normal-sized models prefer `oneshot`.
