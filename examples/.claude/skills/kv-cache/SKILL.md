---
name: kv-cache
description: >
  Generate a working KV cache quantization example script (FP8 KV cache, optionally
  combined with FP8 weights/activations) and save a compressed-tensors checkpoint.
  Triggers on: "kv cache", "kv-cache", "fp8 kv", "quantize kv cache", "kv cache quantization", "per-head kv".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *)]
---

# Write KV Cache Example

Generate a working Python example script that quantizes a model's KV cache to FP8
(optionally alongside FP8 weights and activations) and saves a compressed-tensors
checkpoint.

KV cache quantization shrinks the per-token cache, which is the dominant memory
cost at long context / large batch. It requires calibration data to fit the KV
scales.

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
2. **Scope** — choose one:
   - **KV cache only** — leave weights/activations in full precision, quantize
     just the cache.
   - **FP8 weights + activations + KV cache** — full FP8 (recommended for serving).
3. **Strategy** — `tensor` (per-tensor scales, default) or `tensor_group` /
   per-head for finer granularity.
4. **Model type** — dense, MoE, or multimodal.

KV cache quantization always requires a calibration dataset.

## Step 2 — Choose the template

### FP8 weights + activations + FP8 KV cache (recommended)

```python
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot

MODEL_ID = "<MODEL_ID>"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

# The `kv_cache_scheme` block is what quantizes the cache. Here it is paired with
# FP8 weights + activations; drop the `config_groups` block to quantize the cache
# only.
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""

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

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-KV"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

### KV cache only

Remove the `config_groups` block from the recipe and keep just `kv_cache_scheme`
under `QuantizationModifier`. Weights and activations stay in full precision.

### Per-head KV scales

For finer granularity, set `strategy: tensor_group` (per-head) in
`kv_cache_scheme`. See `examples/quantization_kv_cache/llama3_fp8_head_kv_example.py`.

## Step 3 — Apply model-type adjustments

### Dense models
`ignore=["lm_head"]` is sufficient.

### MoE models
Add gate/router layers to `ignore` (e.g. Qwen MoE: `"re:.*mlp\.gate$"`,
`"re:.*shared_expert_gate.*"`).

### Multimodal (vision / audio)
- Use `AutoProcessor` instead of `AutoTokenizer`
- Add the vision/audio towers to `ignore`
  (`"re:.*vision_tower.*"`, `"re:.*audio_tower.*"`)

## Step 4 — Write the file

Place the file under `examples/quantization_kv_cache/`.

Name the file `{model_name_slug}_fp8_kv_example.py` (e.g. `llama3_fp8_kv_example.py`).

Run `make style` after writing the file.

## Notes

- KV cache quantization requires calibration; `oneshot` with no `dataset` will not
  fit the cache scales.
- Pairing FP8 KV with FP8 weights+activations is the common serving config; the
  cache can also be quantized on its own.
- Per-head (`tensor_group`) scales can recover accuracy on models sensitive to a
  single per-tensor cache scale.
