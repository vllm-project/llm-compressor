## Qwen3.8-Flash-Next NVFP4 Example

Qwen3.8-Flash-Next is a sparse MoE model that requires `transformers >= 5.17.0`. This example quantizes the model to NVFP4 using calibration data and the same single-scheme MoE recipe used for the other Qwen3.8 MoE variants.

### Code Walkthrough

1. Load model
2. Configure quantization recipe
3. Apply quantization
4. Save to disk

### 1. Load Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils import load_context

# NOTE: This example requires transformers >= 5.17.0
MODEL_ID = "Qwen/Qwen3.8-Flash-Next"

with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Configure Quantization Recipe

We quantize the linear layers to NVFP4 and keep `lm_head`, the MoE router (`mlp.gate`), and shared expert gates at full precision.

```python
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:.*mlp.gate$",
        "re:.*shared_expert_gate.*",
    ],
)
```

### 3. Apply Quantization

```python
from llmcompressor import oneshot

oneshot(
    model=model,
    processor=tokenizer,
    recipe=recipe,
    dataset="perfectblend",
    splits="train[:512]",
    max_seq_length=2048,
    num_calibration_samples=512,
    pipeline="sequential",
)
```

### 4. Save to Disk

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

### Pre-quantized Checkpoint

If you do not need to run quantization yourself, the Red Hat AI team has published the resulting NVFP4 checkpoint as [`RedHatAI/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RedHatAI/Qwen3.8-Flash-Next-NVFP4). It can be loaded directly in vLLM:

```python
from vllm import LLM

model = LLM("RedHatAI/Qwen3.8-Flash-Next-NVFP4")
output = model.generate("My name is")
```
