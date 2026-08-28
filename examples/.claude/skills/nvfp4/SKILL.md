---
name: nvfp4
description: >
  Generate a working NVFP4 (W4A4) quantization example script and save a compressed-tensors checkpoint.
  Triggers on: "nvfp4", "NVFP4", "fp4", "nvfp4 example", "quantize to nvfp4", "w4a4".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *), WebFetch]
---

# Write NVFP4 Example

Generate a working Python example script that quantizes a model to an NVFP4 scheme and saves a compressed-tensors checkpoint.

## Shared Documentation

Read `.claude/skills/shared_quantization.md` for common steps on gathering model information, applying model-type adjustments (dense, MoE, multimodal), GPTQ, transforms, and calibration dataset configuration.

## Step 1 — Gather information

Follow the shared documentation for gathering model information, GPTQ, transforms, and calibration dataset configuration.

**IMPORTANT:** NVFP4 is a W4A4 quantization scheme with:
- Weights: fp4 with per-group-16 scaling
- Activations: fp4 with calibrated global scale
- **Requires calibration dataset** for both weight and activation quantization — always use the shared `oneshot_with_data.py` template
- `model_free_ptq` is **NOT supported** — NVFP4 uses the `oneshot` path only

If the user specifically requests `model_free_ptq`, inform them it's not available for NVFP4 and proceed with the `oneshot` approach.

## Step 2 — Use the oneshot template (only path for NVFP4)

Read the shared template at `.claude/skills/templates/oneshot_with_data.py` and use it as the starting point. Set `scheme="NVFP4"`.

Follow the shared documentation to apply GPTQ and/or transform modifications to the recipe if requested.

Apply the model-type adjustments from the shared documentation before writing the final file.

## Step 3 — Apply model-type adjustments

Apply the model-type adjustments documented in `.claude/skills/shared_quantization.md`.

**Note:** For MoE models, the pipeline automatically handles expert calibration via `CalibrationAfmoeMoE` module — no manual intervention needed.

## Step 4 — Write the file

Place the file in `examples/quantization_w4a4_fp4/`.

Name the file `{model_name_slug}_nvfp4.py` (e.g. `llama3_nvfp4.py`, `gemma4_nvfp4.py`).

Run `make style` after writing the file.

## Notes
- NVFP4 **requires** calibration data — unlike FP8 schemes, you cannot use `oneshot(model=model, recipe=recipe)` without a dataset.
- `model_free_ptq` is **not supported** for NVFP4 — always use the `oneshot` path with calibration data.
- NVFP4 targets NVIDIA hardware for W4A4 quantization with per-group weight scaling and calibrated activation scaling.
- The `oneshot` call must include `dataset`, `max_seq_length`, and `num_calibration_samples` parameters.
- `save_compressed=True` is optional — the checkpoint saves in compressed-tensors format either way. Omit unless explicitly requested.
- For MoE models, expert calibration is handled automatically by the `CalibrationAfmoeMoE` module during the calibration phase.
