# `fp8` Weight and Activation Quantization

`llm-compressor` supports quantizing weights and activations to `fp8` for memory savings and inference acceleration with `vllm`

> `fp8` compuation is supported on Nvidia GPUs with compute capability > 8.9 (Ada Lovelace, Hopper).

## Installation

To get started, install:

```bash
pip install llmcompressor==0.1.0
```

## Quickstart

The example includes an end-to-end script for applying the quantization algorithm.

```bash
python3 llama3_example.py
```

The resulting model `Meta-Llama-3.1-8B-Instruct-FP8-Dynamic` is ready to be loaded into vLLM.

## Code Walkthough

Now, we will step though the code in the example. There are three steps:
1) Load model
2) Apply quantization
3) Evaluate accuracy in vLLM

### 1) Load Model

Load the model using `SparseAutoModelForCausalLM`, which is a wrapper around `AutoModelForCausalLM` for saving and loading quantized models.

```python
from llmcompressor.transformers import SparseAutoModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model = SparseAutoModelForCausalLM.from_pretrained(
  MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2) Apply Quantization

For `fp8` quantization, we can recover accuracy with simple PTQ quantization.

We recommend targeting all `Linear` layers using the `FP8_Dynamic` scheme, which uses:
- Static, per-channel quantization on the weights
- Dynamic, per-token quantization on the activations

Since simple PTQ does not require data for weight quantization and the activations are quantized dynamically, we do not need any calibration data for this quantization flow.

```python
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization to run with the FP8_DYNAMIC scheme.
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

We have successfully created an `fp8` model!

### 3) Evaluate Accuracy

Neural Magic's fork of `lm-evaluation-harness` implements the evaluation strategy used by Meta in the Llama3.1 launch. You can install this branch from source below:

```bash
pip install vllm
pip install git+https://github.com/neuralmagic/lm-evaluation-harness.git@a0e54e5f1a0a52abaedced474854ae2ce4e68ded
```

We can now load and run in vLLM:
```python
from vllm import LLM
model = LLM("./Meta-Llama-3.1-8B-Instruct-FP8-Dynamic")
model.generate("Hello my name is")
```

We can evaluate accuracy with `lm_eval`:
> Note: quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.


```bash
MODEL=$PWD/Meta-Llama-3.1-8B-Instruct-FP8-Dynamic 
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks gsm8k_cot_llama_3.1_instruct \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --num_fewshot 8 \
  --batch_size auto
```

We can see the resulting scores look good!

```bash
|           Tasks            |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|----------------------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k_cot_llama_3.1_instruct|      3|flexible-extract|     8|exact_match|↑  |0.8279|±  |0.0104|
|                            |       |strict-match    |     8|exact_match|↑  |0.8203|±  |0.0106|
```

### Questions or Feature Request?

Please open up an issue on `vllm-project/llm-compressor`
