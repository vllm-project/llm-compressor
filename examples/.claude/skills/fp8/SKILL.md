---
name: fp8
description: >
  Generate a working FP8 quantization example script and save a compressed-tensors checkpoint.
  Triggers on: "fp8", "FP8_DYNAMIC", "FP8_BLOCK", "MXFP8", "fp8 example", "quantize to fp8".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *), WebFetch]
---

# Write FP8 Example

Generate a working Python example script that quantizes a model to an FP8 scheme and saves a compressed-tensors checkpoint.

## Step 1 — Gather information

Read the shared documentation at `.claude/skills/shared_quantization.md` for common model information gathering steps.

In addition to the shared information, ask the user (or infer from context) for:

1. **Scheme variant** — choose one:
   - `FP8_BLOCK` — weights fp8 with 128x128 block scaling, activations dynamic. No calibration. Best throughput on Hopper/Blackwell. **Note:** may not be a good choice if the model's weight shapes are not compatible with 128x128 block tiling (e.g. small or irregular hidden dims).
   - `FP8_DYNAMIC` — weights fp8 per-channel, activations fp8 dynamic per-token. No calibration. Broadest hardware support (Ampere+). Use when FP8_BLOCK is not suitable.
   - `MXFP8` — weights and activations in MX fp8 format. No calibration. AMD MI300X target.
2. **Use `model_free_ptq`?** — use `model_free_ptq` if (a) the model does **not** have a definition in the `transformers` library (custom architectures must go through this path), or (b) the model is extremely large (~1TB+) and you want to quantize directly from safetensors files without loading the full model. Otherwise use `oneshot`.

## Templates

Templates are located in `.claude/skills/fp8/templates/`:

- `oneshot.py` — dense-model base template for `oneshot` with `QuantizationModifier`
- `model_free_ptq.py` — template for `model_free_ptq` (no transformers class, or ~1TB+ models)

## Step 2 — Choose the template

### `oneshot` with `QuantizationModifier` (standard path)

Read `templates/oneshot.py` and use it as the starting point. Apply the model-type adjustments from the shared documentation (`.claude/skills/shared_quantization.md`) before writing the final file.

## Step 3 — Apply model-type adjustments

Apply the model-type adjustments documented in `.claude/skills/shared_quantization.md`.

## Step 4 — `model_free_ptq` (no transformers model definition, or very large models ~1TB+)

Read `templates/model_free_ptq.py` and use it as the starting point. Apply the same `ignore` adjustments from the shared documentation (gate/router layers, vision tower layers) before writing the final file.

## Step 5 — Write the file

Place the file in the appropriate directory under `examples/`:
- `quantization_w8a8_fp8/` for FP8_DYNAMIC or FP8_BLOCK (oneshot path)
- `quantization_w8a8_mxfp8/` for MXFP8
- `model_free_ptq/` when using `model_free_ptq()`

Name the file `{model_name_slug}_example.py` (e.g. `llama3_example.py`, `gemma4_example.py`).

Run `make style` after writing the file.

## Notes
- `FP8_BLOCK` is preferred for Hopper/Blackwell throughput, but check that the model's weight shapes are compatible with 128x128 block tiling before choosing it.
- `FP8_DYNAMIC` is the fallback when FP8_BLOCK is not suitable — broad hardware support (Ampere+), no calibration required.
- `MXFP8` targets AMD MI300X.
- Neither scheme requires a calibration dataset; `oneshot(model=model, recipe=recipe)` with no `dataset` argument is correct.
- Use `model_free_ptq` when the model has no transformers class definition, or when the model is ~1TB+ and you want to quantize directly from safetensors without loading the full model.
- `save_compressed=True` is optional — the checkpoint saves in compressed-tensors format either way. Omit unless explicitly requested.
