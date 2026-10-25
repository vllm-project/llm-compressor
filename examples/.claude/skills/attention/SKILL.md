---
name: attention
description: >
  Generate a working attention quantization example script (FP8 attention activations)
  and save a compressed-tensors checkpoint.
  Triggers on: "attention quantization", "quantize attention", "fp8 attention", "attention fp8", "qkv quantization".
allowed-tools: [Read, Write, Glob, Bash(make style), Bash(ls *), Bash(find *)]
---

# Write Attention Example

Generate a working Python example script that quantizes the attention activations
of a model to FP8 and saves a compressed-tensors checkpoint.

Attention quantization targets the model's attention module directly (not the
`Linear` layers), so the target is the **attention class name** for the model
architecture. It requires calibration data.

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
2. **Attention module class** — the model-specific attention class to target, e.g.
   - Llama: `LlamaAttention`
   - Qwen2 / Qwen2.5: `Qwen2Attention`
   - Mistral: `MistralAttention`
   Find it with `print(type(model.model.layers[0].self_attn).__name__)`.
3. **Model type** — dense, MoE, or multimodal.

Attention quantization always requires a calibration dataset.

## Step 2 — Choose the template

```python
from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "<MODEL_ID>"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

NUM_CALIBRATION_SAMPLES = 512
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

# Target the attention module class (NOT "Linear"). Replace "<ATTENTION_CLASS>"
# with the model's attention class, e.g. "LlamaAttention" / "Qwen2Attention".
recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=["<ATTENTION_CLASS>"],
            input_activations=QuantizationArgs(
                num_bits=8, type="float", strategy="tensor"
            ),
        )
    }
)

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

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-attention-fp8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Step 3 — Apply model-type adjustments

### Dense models
Set `targets` to the model's attention class name (see Step 1).

### Combining with weight/activation or KV-cache quantization
Add more entries to `config_groups` (e.g. a `Linear` group for FP8 weights, or a
`kv_cache_scheme`) to quantize attention together with the rest of the model.

### Multimodal (vision / audio)
Use `AutoProcessor`, and target the language-model attention class (the vision
tower has its own attention class — usually left in full precision).

## Step 4 — Write the file

Place the file under `examples/quantization_attention/`.

Name the file `{model_name_slug}_attention.py` (e.g. `llama3_attention.py`).

Run `make style` after writing the file.

## Notes

- The target is the **attention module class**, not `Linear` — using `"Linear"`
  here quantizes the projections, not the attention compute.
- Attention quantization is most useful combined with KV-cache quantization for
  long-context serving.
- The attention class name is architecture-specific; always confirm it from the
  loaded model rather than assuming.
