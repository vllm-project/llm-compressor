# Shared Quantization Skill Documentation

This file contains common logic shared between FP8 and NVFP4 quantization skills.

## Gathering Model Information

When generating a quantization example, collect the following information:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)

2. **Model type** — dense, MoE, or multimodal (vision/audio)

3. **Architecture class (VLM models only, REQUIRED)** — for any multimodal model, fetch the model's `config.json` from HuggingFace to get the exact architecture class before writing the file:
   - Fetch `https://huggingface.co/{MODEL_ID}/raw/main/config.json` and read the `architectures` field (e.g. `["Llama4ForConditionalGeneration"]`)
   - Use that class directly in the import and `from_pretrained` call instead of `AutoModelForCausalLM`
   - To check if the class is in standard `transformers`, run: `python -c "from transformers import <ClassName>"`. If it imports successfully, no `trust_remote_code` needed. If it fails with `ImportError`, add `trust_remote_code=True` to both `from_pretrained` calls.
   - Do **not** use `AutoModelForCausalLM` for VLM models — always use the specific class
   - MoE models can use `AutoModelForCausalLM` like dense models

4. **Gate/router layer names (MoE models only, REQUIRED)** — for any MoE model you **must** determine the exact names of the gate/router layers before writing the file. Do one of the following:
   - Ask the user: *"What are the gating/routing layer names in this model? (e.g. `mlp.gate`, `router`)"*
   - Or inspect the model config: fetch `https://huggingface.co/{MODEL_ID}/raw/main/config.json`, search HuggingFace, or look at existing examples in `examples/` for architecture details to derive the correct regex pattern.
   - Do **not** write the file until the gate/router layer pattern is confirmed. These layers produce logits that control expert routing — quantizing them degrades routing decisions and causes accuracy loss.
   - Once you have the layer name, construct the regex as follows:
     - Escape any literal dots: `mlp.gate` → `mlp\.gate`
     - Anchor with `$` to prevent partial matches on layers sharing the same prefix (e.g. `mlp\.gate$` matches `mlp.gate` but not `mlp.gate_proj`)
     - Prepend `.*` to match at any depth: `"re:.*mlp\.gate$"`
     - **Always use a raw string (`r"..."`) for any ignore pattern containing a backslash** to avoid Python `SyntaxWarning: invalid escape sequence`

## Model-Type Adjustments

**Rule:** Always ignore `lm_head`, any vision tower layers, and any gating/routing layers. The exact regex patterns must be derived from the model's actual architecture. All other layers (e.g. `model.embed_tokens`, attention layers, projectors) are only added to `ignore` when the user explicitly requests it.

### Dense models
`ignore=["lm_head"]` is sufficient.

### MoE models
**Required:** Always add gate/router layers to `ignore`. These control expert routing — quantizing them causes routing degradation. The exact pattern must come from the model's actual architecture.

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
- Trinity: `"re:.*mlp.router.*"`

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
  - Vision: `["re:.*visual.*", ".*vision_tower.*"]`
  - Audio: `"re:audio.*"` (matches `audio_tower.*` etc.)

## GPTQ

GPTQModifier can be used in place of QuantizationModifier for better accuracy at the cost of longer calibration time. It works with any quantization scheme (NVFP4, FP8, etc.).

Ask the user if they want to use GPTQ (default: No, use QuantizationModifier for faster calibration).

**If using GPTQ:** Replace the recipe import and definition:
```python
from llmcompressor.modifiers.gptq import GPTQModifier

recipe = GPTQModifier(
    targets="Linear",
    scheme="<SCHEME>",
    ignore=["lm_head"],
    actorder="static",  # or None if user prefers no specific ordering
    dampening_frac=0.01,  # can be adjusted if Hessian inversion issues occur
    offload_hessians=False,  # set to True for models ≥1TB
    block_size=128,  # user can adjust if desired
)
```
Update the save directory suffix to include `-GPTQ` (e.g. `-NVFP4-GPTQ`, `-FP8_DYNAMIC-GPTQ`).

Use smart defaults and inform the user of the configuration. Don't prompt for each parameter unless the user explicitly wants to customize:

1. **Activation ordering (`actorder`)** — Controls the order in which weight columns are quantized
   - Default: `"static"` (recommended for best accuracy recovery with no runtime cost)
   - User can optionally set to `None` for no specific ordering
2. **Offload Hessians (`offload_hessians`)** — Whether to offload Hessian matrices to CPU during quantization
   - **Checkpoint size estimation:** For fp16 models, approximate checkpoint size = (total parameters × 2 bytes) / 1024^4 TB
     - Example: 70B params → ~0.13 TB, 405B params → ~0.75 TB, 500B params → ~0.93 TB
   - Auto-suggest `True` for models with checkpoint size ≥1TB (reduces GPU memory usage at cost of speed)
     - This typically means models with **500B+ parameters** in fp16
   - Auto-suggest `False` for models <1TB (faster, requires more GPU memory)
   - User can override based on their specific memory constraints
3. **Dampening fraction (`dampening_frac`)** — Hessian dampening for numerical stability
   - Default: `0.01`
   - User can adjust if they encounter Hessian inversion issues during quantization
4. **Block size (`block_size`)** — Number of columns to compress in one pass
   - Default: `128`
   - User can adjust if desired

After determining configuration, inform the user: "Using GPTQ with: actorder='static', dampening_frac=0.01, block_size=128, offload_hessians=[True/False based on model size]. These can be adjusted if needed."

## Custom Quantization Parameters

By default, use a preset scheme string (e.g. `scheme="NVFP4"`, `scheme="FP8_DYNAMIC"`). If the user wants more control, they can specify custom quantization parameters using `config_groups` instead of `scheme`. These two options are mutually exclusive.

`config_groups` maps group names to `QuantizationScheme` objects (or plain dicts), each specifying `targets`, `weights`, and optionally `input_activations`:

```python
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=8,
                type="int",
                strategy="channel",
                symmetric=True,
                dynamic=False,
            ),
            input_activations=QuantizationArgs(
                num_bits=8,
                type="int",
                strategy="token",
                symmetric=True,
                dynamic=True,
            ),
        ),
    },
    ignore=["lm_head"],
)
```

Plain dicts work too — Pydantic validates them automatically:
```python
recipe = QuantizationModifier(
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            },
        },
    },
    ignore=["lm_head"],
)
```

Key `QuantizationArgs` fields:
- `num_bits` — bit depth (e.g. 4, 8)
- `type` — `"int"` or `"float"`
- `symmetric` — whether scale is symmetric about zero
- `strategy` — one of `"tensor"`, `"channel"`, `"group"`, `"block"`, `"token"`, `"tensor_group"`
- `group_size` — group length for `"group"` / `"tensor_group"` strategy
- `block_structure` — 2D block dims like `[128, 128]` for `"block"` strategy
- `dynamic` — `True` for fully dynamic, `False` for static
- `observer` — observer algorithm for calibration (default: `None`, which falls back to min-max for non-dynamic quantization):
  - `"minmax"` — running min/max
  - `"memoryless_minmax"` — min/max without history, uses only the current batch
  - `"static_minmax"` — computes scale once from the first batch, then freezes
  - `"mse"` — minimizes mean squared error between quantized and original values
  - `"memoryless_mse"` — MSE without history
  - `"imatrix_mse"` — MSE weighted by importance matrix (use with `IMatrixGatherer` transform)

Custom parameters work with `GPTQModifier` the same way — replace `QuantizationModifier` with `GPTQModifier` and use `config_groups`.

**Non-uniform / mixed-precision:** Multiple config groups allow applying different quantization to different layers:
```python
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, NVFP4

recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[r"re:.*self_attn\..*"],
            **FP8_BLOCK,
        ),
        "experts": QuantizationScheme(
            targets=[r"re:.*mlp.*"],
            **NVFP4,
        ),
    },
    ignore=["lm_head"],
)
```

## Transforms

A transform can improve quantization accuracy by redistributing quantization difficulty before the quantization step. Transforms work with any quantization scheme and compose with either `QuantizationModifier` or `GPTQModifier`.

Ask the user if they want to apply a transform (default: No):

- **SmoothQuant** — Migrates quantization difficulty from activations to weights via channel-wise scaling. Key parameter: `smoothing_strength` (default: `0.5`, range 0–1; higher values shift more difficulty to weights).
- **AWQ** — Activation-aware Weight Quantization. Identifies salient weights based on activation magnitudes and applies channel-wise scaling to preserve them. Key parameter: `duo_scaling` (default: `"both"`).

**If using a transform:** Wrap the recipe in a list with the transform modifier first, followed by the quantization modifier. Add the corresponding import.

With SmoothQuant:
```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.5),
    QuantizationModifier(
        targets="Linear",
        scheme="<SCHEME>",
        ignore=["lm_head"],
    ),
]
```

With AWQ:
```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

recipe = [
    AWQModifier(duo_scaling="both"),
    QuantizationModifier(
        targets="Linear",
        scheme="<SCHEME>",
        ignore=["lm_head"],
    ),
]
```

Transforms compose with GPTQ as well — replace `QuantizationModifier` with `GPTQModifier` in the recipe list and update the save directory suffix accordingly (e.g. `-NVFP4-GPTQ-SmoothQuant`).

Update the save directory suffix to include the transform name (e.g. `-NVFP4-SmoothQuant`, `-FP8_DYNAMIC-AWQ`).

## Calibration Dataset

Some configurations require a calibration dataset:
- NVFP4 **always** requires calibration data
- GPTQ **always** requires calibration data (regardless of scheme)
- Transforms (AWQ, SmoothQuant) **always** require calibration data
- FP8 with plain QuantizationModifier (no GPTQ, no transform) does **not** require calibration data

When calibration data is needed, **start with the default template** — the manual `load_dataset` → `preprocess` → `tokenize` block using the template's default dataset (`HuggingFaceH4/ultrachat_200k`, split `train_sft`, chat template applied to a `messages` column). Do not ask the user to pick a dataset up front.

**After** wiring up the default template, ask the user a single follow-up: *"The example uses the default `ultrachat_200k` dataset with manual preprocessing. Would you like to swap it for a prebaked (registered) dataset instead?"* Offer a few common registered datasets as selectable options, and let them type any other value:

- Keep the default (`ultrachat_200k`, manual processing) — **recommended default**
- `perfectblend` — general-purpose text blend
- `open_platypus` — instruction
- Or the user can type any other registered dataset name.

Registered datasets are passed to `oneshot` **by name** and handle loading, preprocessing, and tokenization internally. The full list of registered names lives in `src/llmcompressor/transformers/data/` (each file has a `@TextGenerationDataset.register(name=...)` decorator) — point the user there rather than enumerating every option.

Also determine:
- **Number of calibration samples** — how many samples to use (default: `256` for QuantizationModifier alone, `512` for GPTQModifier or when using a transform; MoE models may benefit from more samples)
- **Max sequence length** — maximum sequence length for tokenization (default: `2048`)

**Default behavior:** If the user doesn't respond or has no preference, keep the default template (manual `ultrachat_200k` block).

### Writing the dataset code

Use the shared template at `.claude/skills/templates/oneshot_with_data.py` as the starting point. Choose the dataset code based on what the user selected — **never ask the user to edit or delete code themselves**:

- **Default (keep the template's manual block):** keep the manual `load_dataset` / `preprocess` / `tokenize` block as-is. It loads `HuggingFaceH4/ultrachat_200k` (split `train_sft`) and applies the chat template to the `messages` column. Pass the processed dataset object to `oneshot` as `dataset=ds`. If the user asks for a *different* HuggingFace dataset / local files with manual processing, keep this block and wire in their values:
  1. Set `DATASET_ID` and `DATASET_SPLIT` to the user's values (template defaults: `HuggingFaceH4/ultrachat_200k` / `train_sft`).
  2. Adjust the `preprocess` function so each example produces a `"text"` field matching the dataset's actual columns (the template default applies the chat template to a `messages` column).
  3. Keep the `tokenize` function and the `ds.map(...)` calls as-is.
- **Swap to a registered dataset (by name):** if the user opts to swap, omit the manual `load_dataset` / `preprocess` / `tokenize` block entirely and pass the name directly, e.g.:
  ```python
  oneshot(
      model=model,
      dataset="perfectblend",
    splits="train[:512]",
      recipe=recipe,
      max_seq_length=MAX_SEQUENCE_LENGTH,
      num_calibration_samples=NUM_CALIBRATION_SAMPLES,
  )
  ```

## Sample Generation

Check the model's parameter count from its name or by fetching `config.json` (look for a size indicator in the model ID such as `70B`, `72B`, `405B`). If the model exceeds **70B parameters**, omit the entire sample generation block (from `dispatch_model` through the closing `print`) to avoid OOM.

## File Naming and Styling

After writing the file, always run `make style` to format it correctly.
