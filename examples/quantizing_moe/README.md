# Quantizing Mixture of Experts (MoE) models

These examples demonstrate how to quantize MoE models using `llm-compressor`. We'll walk through the GLM-4.7 example which applies AWQ quantization to create a W4A16 (4-bit weights, 16-bit activations) model.

## Installation

To get started, install:

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## End-to-End Example: Quantizing GLM-4.7

You can run the complete example with:

```bash
python3 glm4_7_example.py
```

This example demonstrates quantizing the `zai-org/GLM-4.7` MoE model using AWQ (Activation-aware Weight Quantization) to 4-bit precision. The process automatically handles MoE-specific calibration requirements.

### Step 1: Load the Model and Tokenizer

First, load the GLM-4.7 model and its tokenizer from the Hugging Face Hub. The `load_context` context is responsible for ensuring that moe modules load in a way such that they can be calibrated properly. For more information on supporting new MoE architectures, see [MoE Support Guide](../../docs/developer-tutorials/add-moe-support.md).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.utils import load_context

model_id = "zai-org/GLM-4.7"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### Step 2: Prepare the Calibration Dataset

Load and preprocess a calibration dataset. In this example, we use `ultrachat_200k`:

```python
from datasets import load_dataset

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load and shuffle the dataset
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# Apply chat template
def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

# Tokenize
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

**Note**: 512 calibration samples is a good starting point. Increasing the number of samples can improve quantization accuracy.

### Step 3: Configure the Quantization Recipe

Define which layers to quantize and which to ignore. GLM-4.7 has dense layers at the beginning that should be excluded:

```python
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

moe_ignores = [
    # Layers 0-2: Dense layers - ignore entire layers
    "model.layers.0.*",
    "model.layers.1.*",
    "model.layers.2.*",
    # Ignore the output head
    "lm_head",
]

# Configure AWQ with W4A16 (4-bit weights, 16-bit activations)
recipe = [
    AWQModifier(),
    QuantizationModifier(targets="Linear", scheme="W4A16", ignore=moe_ignores)
]
```

**Why ignore these layers?**
- Layers 0-2 are dense (non-MoE) layers that may be sensitive to aggressive quantization
- The `lm_head` (language model head) is typically kept at higher precision for better output quality

### Step 4: Run Quantization with `oneshot`

Apply the quantization recipe using the `oneshot` method:

```python
from llmcompressor import oneshot

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
```

The `oneshot` method:
- Calibrates the quantization parameters using the provided dataset
- Applies AWQ to quantize weights to 4-bit precision
- Automatically uses the calibration-friendly MoE definition to ensure all experts are properly calibrated

### Step 5: Save the Quantized Model

Save the compressed model to disk:

```python
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

The model will be saved in a compressed format with 4-bit weights, ready for vLLM inference.
