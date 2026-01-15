# Priority Model Examples for FP8 Quantization

Below are examples for FP8 quantization for Llama4, Qwen3, Kimi K2, and Mistral Large 3.

## Llama4

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Load model
2. Configure quantization algorithm and scheme
3. Apply quantization
4. Confirm generations of the quantized model look sane
5. Save to disk in compressed-tensors format

### 1. Load Model

Load the model using `AutoModelForCausalLM`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Configure the Quantization Algorithm and Scheme

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*lm_head",
        "re:.*self_attn",
        "re:.*router",
        "re:.*vision_model.*",
        "re:.*multi_modal_projector.*",
        "Llama4TextAttention",
    ],
)
```

### 3. Apply Quantization

```python
oneshot(model=model, recipe=recipe)
```

### 4. Confirm Generations of the Quantized Model Look Sane

```python
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")
```

### 5. Save to Disk in Compressed-Tensors Format

```python
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-BLOCK"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

## Qwen3

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Load model
2. Configure quantization algorithm and scheme
3. Apply quantization
4. Save to disk in compressed-tensors format

### 1. Load Model

```python
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE: Requires a minimum of transformers 4.57.0

MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

### 2. Configure Quantization Algorithm and Scheme

In this case, we are doing the following:
 * quantize the weights to fp8 with channel-wise quantization
 * quantize the activations to fp8 with dynamic token activations

NOTE: Only datafree quantization is supported for Qwen3-VL-MoE currently

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
    ],
)
```

### 3. Apply Quantization

```python
oneshot(model=model, recipe=recipe)
```

### 4. Save to Disk in Compressed-Tensors Format

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-DYNAMIC"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

## Kimi-K2

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Load model
2. Apply quantization

### 1. Load Model

```python
from llmcompressor import model_free_ptq

MODEL_ID = "unsloth/Kimi-K2-Thinking-BF16"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"
```

### 2. Apply Quantization

Once quantized, the model is saved. This uses compressed-tensors to the SAVE_DIR.

```python
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*gate$",
        "lm_head",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "model.embed_tokens",
    ],
    max_workers=15,
    device="cuda:0",
)
```

## Mistral Large 3

### Code Walkthrough

#### Prerequisite: Script

```python
"""
NOTE: Please run the following script before using `model_free_ptq`

This script is used to reindex the safetensors files of a model such that all fused
modules (gate_up, qkv) are in the same safetensors file. This is required by
model_free_ptq for microscale schemes (NVFP4A16, MXFP4A16)

llmcompressor.reindex_fused_weights \
    mistralai/Mistral-Large-3-675B-Instruct-2512-BF16 \
    Mistral-Large-3-675B-Instruct-2512-BF16-reindexed \
    --num_workers=10
"""
```

Let's walk through the main steps of the quantization process:
1. Load model
2. Apply quantization
3. Modify ignore list

### 1. Load Model

```python
from llmcompressor import model_free_ptq

MODEL_ID = "mistralai/Mistral-Large-3-675B-Instruct-2512-BF16"
REINDEX_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-reindexed"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16"
```

### 2. Apply Quantization

```python
model_free_ptq(
    REINDEX_DIR,
    SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=[
        "tok_embeddings",  # embeddings
        "re:patch_merger.*",  # patch merger
        "re:vision_encoder.*",  # vision tower
        "re:vision_language_adapter.*",  # vision adapter
        "re:.*wkv_a_with_mqa$",  # non divisible
        "re:.*wq_a$",  # fused with wkv_a_with_mqa
        "re:.*gate$",  # gate layers
        "output",  # lm head
    ],
    max_workers=10,
    device="cuda:0",
)
```

### 3. Modify Ignore List

```python
# "ignore": [
#   "model.embed_tokens",
#   "re:patch_merger.*",
#   "re:vision_encoder.*",
#   "re:vision_language_adapter.*",
#   "re:.*kv_a_proj_with_mqa$",
#   "re:.*q_a_proj$",
#   "re:.*gate$",
#   "lm_head"
# ],
```

