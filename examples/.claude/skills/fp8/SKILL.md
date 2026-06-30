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

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
2. **Scheme variant** — choose one:
   - `FP8_BLOCK` — weights fp8 with 128x128 block scaling, activations dynamic. No calibration. Best throughput on Hopper/Blackwell. **Note:** may not be a good choice if the model's weight shapes are not compatible with 128x128 block tiling (e.g. small or irregular hidden dims).
   - `FP8_DYNAMIC` — weights fp8 per-channel, activations fp8 dynamic per-token. No calibration. Broadest hardware support (Ampere+). Use when FP8_BLOCK is not suitable.
   - `MXFP8` — weights and activations in MX fp8 format. No calibration. AMD MI300X target.
3. **Model type** — dense, MoE, or multimodal (vision/audio)
4. **Use `model_free_ptq`?** — use `model_free_ptq` if (a) the model does **not** have a definition in the `transformers` library (custom architectures must go through this path), or (b) the model is extremely large (~1TB+) and you want to quantize directly from safetensors files without loading the full model. Otherwise use `oneshot`.
5. **Architecture class (VLM models only, REQUIRED)** — for any multimodal model, fetch the model's `config.json` from HuggingFace to get the exact architecture class before writing the file:
   - Fetch `https://huggingface.co/{MODEL_ID}/raw/main/config.json` and read the `architectures` field (e.g. `["Llama4ForConditionalGeneration"]`)
   - Use that class directly in the import and `from_pretrained` call instead of `AutoModelForCausalLM`
   - To check if the class is in standard `transformers`, run: `python -c "from transformers import <ClassName>"`. If it imports successfully, no `trust_remote_code` needed. If it fails with `ImportError`, add `trust_remote_code=True` to both `from_pretrained` calls.
   - Do **not** use `AutoModelForCausalLM` for VLM models — always use the specific class
   - MoE models can use `AutoModelForCausalLM` like dense models

6. **Gate/router layer names (MoE models only, REQUIRED)** — for any MoE model you **must** determine the exact names of the gate/router layers before writing the file. Do one of the following:
   - Ask the user: *"What are the gating/routing layer names in this model? (e.g. `mlp.gate`, `router`)"*
   - Or inspect the model config: fetch `https://huggingface.co/{MODEL_ID}/raw/main/config.json`, search HuggingFace, or look at existing examples in `examples/` for architecture details to derive the correct regex pattern.
   - Do **not** write the file until the gate/router layer pattern is confirmed. These layers produce logits that control expert routing — quantizing them degrades routing decisions and causes accuracy loss.
   - Once you have the layer name, construct the regex as follows:
     - Escape any literal dots: `mlp.gate` → `mlp\.gate`
     - Anchor with `$` to prevent partial matches on layers sharing the same prefix (e.g. `mlp\.gate$` matches `mlp.gate` but not `mlp.gate_proj`)
     - Prepend `.*` to match at any depth: `"re:.*mlp\.gate$"`

## Step 2 — Choose the template

### `oneshot` with `QuantizationModifier` (standard path)

**Dense models** — use `AutoModelForCausalLM`:

```python
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "<MODEL_ID>"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="<SCHEME>",  # FP8_DYNAMIC | FP8_BLOCK | MXFP8
    ignore=["lm_head"],  # add model-specific vision tower and gating/routing layers — see Step 3
)

oneshot(model=model, recipe=recipe)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-<SCHEME>"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

**MoE models** — use `AutoModelForCausalLM` like dense models; just add gate/router layers to `ignore` (see Step 3).

**VLM models** — always use the specific architecture class from `config.json`. **Never use `AutoModelForCausalLM` for VLM models.**

```python
from compressed_tensors.offload import dispatch_model
from transformers import SpecificClass, AutoProcessor  # e.g. Llama4ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "<MODEL_ID>"

# Use the exact class from config.json `architectures` field.
# If the class requires trust_remote_code (custom architecture not in standard transformers),
# add trust_remote_code=True to both from_pretrained calls.
with load_context(SpecificClass):
    model = SpecificClass.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="<SCHEME>",  # FP8_DYNAMIC | FP8_BLOCK | MXFP8
    ignore=[
        "lm_head",
        # add model-specific vision tower and gating/routing layers — see Step 3
    ],
)

oneshot(model=model, recipe=recipe)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = processor("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-<SCHEME>"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

### `model_free_ptq` (no transformers model definition, or very large models ~1TB+)

```python
from llmcompressor import model_free_ptq

MODEL_ID = "<MODEL_ID>"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-<SCHEME>"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="<SCHEME>",  # FP8_DYNAMIC | FP8_BLOCK
    ignore=[
        "lm_head",
        # add model-specific vision tower and gating/routing layers — see Step 3
    ],
    max_workers=15,
    device="cuda:0",
)
```

## Step 3 — Apply model-type adjustments

**Rule:** Always ignore `lm_head`, any vision tower layers, and any gating/routing layers. The exact regex patterns must be derived from the model's actual architecture. All other layers (e.g. `model.embed_tokens`, attention layers, projectors) are only added to `ignore` when the user explicitly requests it.

### Dense models
`ignore=["lm_head"]` is sufficient.

### MoE models
**Required:** Always add gate/router layers to `ignore`. These control expert routing — quantizing them causes routing degradation. The exact pattern must come from the model's actual architecture (see Step 1 item 6).

Always wrap the load in `load_context` using `AutoModelForCausalLM`:
```python
from transformers import AutoModelForCausalLM
from llmcompressor.utils import load_context

with load_context(AutoModelForCausalLM):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
```
Common MoE gating patterns (always verify against the model's actual layer names before using):
- Qwen MoE: `"re:.*mlp.gate$"`, `"re:.*shared_expert_gate.*"`
- Llama4 / Gemma4 MoE: `"re:.*router"`
- Laguna-M.1: `"re:.*mlp\.gate$"` (targets `mlp.gate` routing layers; avoids false match on `gate_proj`)

If the gate/router layer is an `nn.Parameter` (not `nn.Linear`) it will not be targeted by `targets="Linear"`, but it should still be explicitly listed in `ignore` to document intent and guard against future changes.
Some deepseek-like architectures use an `attn.indexer` and `attn.indexer.compressor`. These weights are sensitive to quantization and should be ignored.
### Multimodal (vision / audio)
- Use `AutoProcessor` instead of `AutoTokenizer`
- Always fetch the model's `config.json` to get the specific class from the `architectures` field (e.g. `Gemma4ForConditionalGeneration`, `Llama4ForConditionalGeneration`) — **never use `AutoModelForCausalLM` for VLMs**
- Import and use that class directly; wrap in `load_context`:
  ```python
  from transformers import Llama4ForConditionalGeneration  # example
  from llmcompressor.utils import load_context

  with load_context(Llama4ForConditionalGeneration):
      model = Llama4ForConditionalGeneration.from_pretrained(MODEL_ID)
  ```
- Add `trust_remote_code=True` if the class is not in standard transformers
- Always ignore vision tower layers using a single regex that covers the model's actual layer names (verify against the architecture — one pattern is sufficient since regex search matches substrings):
  - Vision: `["re:.*visual.*, ".*vision_tower.*"]`
  - Audio: `"re:audio.*"` (matches `audio_tower.*` etc.)

## Step 4 — Write the file

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
