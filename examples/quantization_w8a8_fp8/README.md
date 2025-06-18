# `fp8` Weight and Activation Quantization

`llmcompressor` supports quantizing weights and activations to `fp8` for memory savings and inference acceleration with `vllm`

> `fp8` compuation is supported on Nvidia GPUs with compute capability > 8.9 (Ada Lovelace, Hopper).

## Installation

To get started, install:

```bash
pip install llmcompressor
```

## Quickstart

The example includes an end-to-end script for applying the quantization algorithm.

```bash
python3 llama3_example.py
```

The resulting model `Meta-Llama-3-8B-Instruct-FP8-Dynamic` is ready to be loaded into vLLM.

## Code Walkthough

Now, we will step though the code in the example. There are three steps:
1) Load model
2) Apply quantization
3) Evaluate accuracy in vLLM

### 1) Load Model

Load the model using `AutoModelForCausalLM`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2) Apply Quantization

For `fp8` quantization, we can recover accuracy with simple PTQ quantization.

We recommend targeting all `Linear` layers using the `FP8_DYNAMIC` scheme, which uses:
- Static, per-channel quantization on the weights
- Dynamic, per-token quantization on the activations

Since simple PTQ does not require data for weight quantization and the activations are quantized dynamically, we do not need any calibration data for this quantization flow.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

We have successfully created an `fp8` model!

### 3) Evaluate Accuracy

Install `vllm` and `lm-evaluation-harness`:

```bash
pip install vllm lm_eval==0.4.3
```

Load and run the model in `vllm`:

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
model.generate("Hello my name is")
```

Evaluate accuracy with `lm_eval` (for example on 250 samples of `gsm8k`):
> Note: quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.

```bash
MODEL=$PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic 
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250
```

We can see the resulting scores look good:

```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
|     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|
```

### Questions or Feature Request?

Please open up an issue on `vllm-project/llm-compressor`
