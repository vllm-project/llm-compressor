## DeepSeek V4 NVFP4 + FP8 Example

This example quantizes DeepSeek-V4-Flash to a mixed-precision format: NVFP4 for MoE expert weights and FP8-Block for attention projection weights.

The full example script can be found [here](../../../examples/quantizing_moe/deepseek_v4_example.py).

### Code Walkthrough

1. Load model
2. Configure quantization recipe
3. Prepare calibration dataset
4. Apply quantization
5. Save to disk

### 1. Load Model

DeepSeek-V4 has an upstream issue where norms are incorrectly loaded in FP32 due to the base checkpoint's `quant_config`. The workaround is to clear `_keep_in_fp32_modules_strict` before loading.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4PreTrainedModel

from llmcompressor.utils import load_context

DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = set()

MODEL_ID = "RedHatAI/DeepSeek-V4-Flash-BF16"

with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Configure Quantization Recipe

DeepSeek-V4 uses a unique attention architecture (CSA with HCA and mHC projections). We quantize:

- **Attention projections** (`q_a_proj`, `q_b_proj`, `kv_proj`, `o_a_proj`, `o_b_proj`, and the compressor indexer) to FP8-Block
- **MoE expert weights** (gate/up/down projections) to NVFP4

```python
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, NVFP4, QuantizationScheme
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[
                r"re:.*attn\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
                r"re:.*attn\.compressor\.indexer\.q_b_proj$",
            ],
            **FP8_BLOCK,
        ),
        "experts": QuantizationScheme(
            targets=[
                r"re:.*mlp\..*(gate|up|down)_proj$",
            ],
            **NVFP4,
        ),
    },
)
```

### 3. Prepare Calibration Dataset

DeepSeek-V4 does not use a standard chat template. The chat format must be encoded manually.

```python
from datasets import load_dataset

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    BOS = "<｜begin▁of▁sentence｜>"
    EOS = "<｜end▁of▁sentence｜>"
    text = BOS
    for message in example["messages"]:
        role = message["role"]
        content = message["content"]
        if role == "system":
            text += content
        elif role == "user":
            text += f"<｜User｜>{content}"
        elif role == "assistant":
            text += f"<｜Assistant｜></think>{content}{EOS}"
    return {"text": text}


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)
```

### 4. Apply Quantization

Due to the size of DeepSeek-V4, `sequential_targets` is used to process one decoder block at a time, keeping memory usage manageable.

```python
from llmcompressor import oneshot

oneshot(
    model=model,
    processor=tokenizer,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["DeepseekV4DecoderLayer"],
    batch_size=1,
)
```

### 5. Save to Disk

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8-BLOCK"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```
