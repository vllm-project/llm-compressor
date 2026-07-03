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

Read `.claude/skills/shared_quantization.md` for common steps on gathering model information and applying model-type adjustments (dense, MoE, multimodal).

## Step 1 — Gather information

Follow the shared documentation for gathering model information.

In addition, ask the user (or use defaults) for:

**Quantization algorithm:**
1. **Use GPTQ?** — GPTQModifier can provide better accuracy than standard QuantizationModifier at the cost of longer calibration time. Ask the user if they want to use GPTQ (default: No, use QuantizationModifier for faster calibration).

**Calibration dataset configuration:**
1. **Dataset ID** — HuggingFace dataset to use for calibration (default: `HuggingFaceH4/ultrachat_200k`)
2. **Dataset split** — which split to use (default: `train_sft`)
3. **Number of calibration samples** — how many samples to use (default: `256` for QuantizationModifier, `512` for GPTQModifier; MoE models may benefit from more samples)
4. **Max sequence length** — maximum sequence length for tokenization (default: `2048`)
5. **Preprocessing** — any custom preprocessing needed beyond the default chat template application

**Default behavior:** If the user doesn't specify dataset preferences, use the template's defaults which are configured for `HuggingFaceH4/ultrachat_200k` with chat template preprocessing.

**IMPORTANT:** NVFP4 is a W4A4 quantization scheme with:
- Weights: fp4 with per-group-16 scaling
- Activations: fp4 with calibrated global scale
- **Requires calibration dataset** for both weight and activation quantization
- `model_free_ptq` is **NOT supported** — NVFP4 uses the `oneshot` path only

If the user specifically requests `model_free_ptq`, inform them it's not available for NVFP4 and proceed with the `oneshot` approach.

## Templates

Templates are located in `.claude/skills/nvfp4/templates/`:

- `oneshot.py` — template for `oneshot` with `QuantizationModifier` including calibration dataset

## Step 2 — Use the oneshot template (only path for NVFP4)

Read `templates/oneshot.py` and use it as the starting point. This template includes:
- Dataset loading and preprocessing (configured for `HuggingFaceH4/ultrachat_200k` by default)
- Chat template preprocessing
- Tokenization pipeline
- `oneshot` call with dataset and calibration parameters
- Uses `QuantizationModifier` with `scheme="NVFP4"` by default

**If using GPTQ:** Replace the recipe import and definition:
```python
from llmcompressor.modifiers.gptq import GPTQModifier

recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head"],
)
```
Also update the save directory suffix to `-NVFP4-GPTQ`.

**Dataset customization:** If the user specified custom dataset preferences in Step 1, modify the template's dataset configuration accordingly:
- Update `DATASET_ID` and `DATASET_SPLIT`
- Adjust `NUM_CALIBRATION_SAMPLES` and `MAX_SEQUENCE_LENGTH`
- Modify the `preprocess()` function if custom preprocessing is needed (the default applies chat template)
- If the dataset doesn't use a `messages` field, adjust the preprocessing logic accordingly

Apply the model-type adjustments from the shared documentation before writing the final file.

## Step 3 — Apply model-type adjustments

Apply the model-type adjustments from the shared documentation (`.claude/skills/shared_quantization.md`).

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
