---
name: autoround
description: >
  Generate a working AutoRound quantization example script (learned weight rounding for
  W4A16 / NVFP4 / MXFP4) and save a compressed-tensors checkpoint.
  Triggers on: "autoround", "auto-round", "auto round", "learned rounding", "sign-SGD rounding".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *)]
---

# Write AutoRound Example

Generate a working Python example script that quantizes a model with the AutoRound
algorithm and saves a compressed-tensors checkpoint.

AutoRound is an alternative weight-quantization algorithm to GPTQ/AWQ: it learns
per-weight rounding via a short optimization, often improving low-bit accuracy. It
applies across schemes (W4A16, NVFP4, MXFP4, FP8). Requires the `auto-round`
package (`pip install auto-round`).

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
2. **Scheme** — `W4A16` (default), `NVFP4`, or `MXFP4`.
3. **iters** — AutoRound optimization steps (200 is a good default; higher = more
   accuracy, slower).
4. **Model type** — dense, MoE, or multimodal.

AutoRound always requires a calibration dataset.

## Step 2 — Choose the template

AutoRound uses its own aligned calibration dataset via `auto_round.calib_dataset`.

```python
from auto_round.calib_dataset import get_dataset
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

MODEL_ID = "<MODEL_ID>"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

# AutoRound-aligned calibration dataset.
ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# Learned 4-bit weight rounding (group 128). Swap scheme for NVFP4 / MXFP4.
recipe = AutoRoundModifier(
    targets="Linear", scheme="W4A16", ignore=["lm_head"], iters=200
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # disable shuffling to get slightly better MMLU score
    shuffle_calibration_samples=False,
)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Step 3 — Apply model-type adjustments

### Dense models
`ignore=["lm_head"]` is sufficient.

### MoE models
Add gate/router layers to `ignore` (e.g. Qwen MoE: `"re:.*mlp\.gate$"`). See
`examples/autoround/ddp/` for a distributed MoE example.

### NVFP4 / MXFP4 with AutoRound
Set `scheme="NVFP4"` or `scheme="MXFP4"`. See
`examples/autoround/quantization_w4a4_fp4/` and
`examples/autoround/quantization_w4a4_mxfp4/`.

### Multimodal (vision / audio)
Use `AutoProcessor` and the model-specific class; add vision/audio towers to
`ignore`. See `examples/autoround/quantization_w4a4_fp4/qwen3_vl_example.py`.

## Step 4 — Write the file

Place the file under `examples/autoround/quantization_<scheme>/`
(e.g. `quantization_w4a16/`, `quantization_w4a4_fp4/`).

Name the file `{model_name_slug}_example.py` (e.g. `llama3_example.py`).

Run `make style` after writing the file.

## Notes

- Requires `pip install auto-round`; calibration uses `auto_round.calib_dataset`.
- `shuffle_calibration_samples=False` is recommended — AutoRound's aligned dataset
  scores slightly better unshuffled.
- AutoRound trades extra calibration time (the `iters` optimization) for better
  low-bit accuracy than plain RTN; compare against GPTQ/AWQ for your model.
