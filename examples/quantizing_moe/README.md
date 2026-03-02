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

First, load the GLM-4.7 model and its tokenizer from the Hugging Face Hub:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modeling.glm4_moe import CalibrationGlm4MoeMoE  # noqa: F401

model_id = "zai-org/GLM-4.7"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

**Important**: The import of `CalibrationGlm4MoeMoE` is crucial for proper MoE calibration. This custom module automatically replaces the original `Glm4MoeMoE` class during calibration to ensure all experts are properly calibrated, even those that wouldn't normally be activated for certain tokens. More details on this can be found in [Quantizing MoEs with a custom definition](#quantizing-moes-with-a-custom-definition).

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
from llmcompressor.modifiers.awq import AWQModifier

moe_ignores = [
    # Layers 0-2: Dense layers - ignore entire layers
    "model.layers.0.*",
    "model.layers.1.*",
    "model.layers.2.*",
    # Ignore the output head
    "lm_head",
]

# Configure AWQ with W4A16 (4-bit weights, 16-bit activations)
recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=moe_ignores)
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

# Quantizing MoEs with a custom definition
Quantizing MoE models with a scheme that requires calibration data (for example, schemes where activations are not dynamic, such as FP8 or INT8 per-tensor activations, or NVFP4), or with an algorithm that requires data (such as GPTQ, AWQ, or AutoRound), requires a calibration-friendly MoE block definition for the model being quantized.

Examples of calibration-friendly definitions can be found in the [modeling folder](https://github.com/vllm-project/llm-compressor/tree/main/src/llmcompressor/modeling). Each definition enables an MoE calibration context by inheriting from the [`MoECalibrationModule` class](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/moe_context.py) and registering the MoE block that should be replaced with a custom definition.

In particular, each model-specific definition includes an updated forward pass that ensures all tokens are routed through all experts during calibration, including experts that would not normally be activated. Only the activated experts contribute to the final output of the MoE block. This behavior ensures proper calibration of all expert layers.

These custom definitions replace the existing MoE implementations during `oneshot` processing. The replacement can be either temporary or permanent; in the temporary case, the original definition is restored after calibration. In the GLM-4.7 example above, the `CalibrationGlm4MoeMoE` custom definition registers a replacement of all `Glm4MoeMoE` instances from the transformers library with the calibration-friendly version. You can see this definition replacement applied in [llmcompressor/modeling/glm4_moe.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/glm4_moe.py).

Without a custom calibration-friendly definition, MoE experts may be calibrated incorrectly, which can result in numerical instability or NaNs.

