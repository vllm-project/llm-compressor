## Kimi-K3 NVFP4 Example

### Overview

Kimi-K3 requires custom modeling files bundled with LLM Compressor, since it is not yet supported in Transformers.
The example below quantizes the model to NVFP4 using calibration data.

The full example script can be found [here](../../../examples/quantizing_moe/kimi_k3_example.py).

### Code Walkthrough

```python
from compressed_tensors.quantization import QuantizationConfig
from transformers import AutoTokenizer

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "moonshotai/Kimi-K3"

# Load quantization config from pretrained and add ignore patterns
# for modules that should not be quantized
qconfig = QuantizationConfig.from_pretrained(MODEL_ID)
qconfig.ignore += [
    "re:.*mlp_res_proj.*",
    "re:.*self_attention_res_proj.*",
    "re:.*routed_expert.*",
    "re:.*output_attn_res_proj.*",
]

# Load model with the modified quantization config
with load_context(KimiK3ForConditionalGeneration):
    model = KimiK3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=qconfig,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


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

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        r"re:.*block_sparse_moe\.gate",
        "re:.*vision_tower.*",
    ],
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```
