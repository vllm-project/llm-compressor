## HY-V3 NVFP4 + FP8 Example

This example quantizes Tencent's HY-V3 (Hy3) sparse MoE model to a mixed-precision format: NVFP4 for MoE expert weights and FP8-Block for attention weights. Due to the model's size, quantization requires multiple GPUs via DDP.

The full example script can be found [here](../../../examples/quantization_w4a4_fp4/hy3_example.py).

### Code Walkthrough

1. Load model with DDP
2. Configure quantization recipe
3. Prepare calibration dataset
4. Apply quantization
5. Save to disk

### 1. Load Model with DDP

HY-V3 is too large to fit on a single GPU for calibration. We use `init_dist()` to initialize distributed quantization and `auto_offload` to stream weights from CPU as needed.

```python
import torch
from compressed_tensors.distributed import init_dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils import load_context

MODEL_ID = "tencent/Hy3"

init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={},
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Configure Quantization Recipe

We quantize:

- **Attention weights** to FP8-Block
- **MoE expert weights** (all MLP layers) to NVFP4
- The `lm_head` is kept at full precision

```python
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, NVFP4, QuantizationScheme
from llmcompressor.modifiers.quantization import QuantizationModifier

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

### 3. Prepare Calibration Dataset

```python
from datasets import load_dataset
from llmcompressor.datasets.utils import get_rank_partition

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
)
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
```

### 4. Apply Quantization

```python
from llmcompressor import oneshot

oneshot(
    model=model,
    processor=tokenizer,
    dataset=ds,
    recipe=recipe,
    batch_size=4,
    shuffle_calibration_samples=False,
)
```

### 5. Save to Disk

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
```
