# `fp8` Weight, Activation, and KV Cache Quantization

`llmcompressor` now supports quantizing weights, activations, and KV cache to `fp8` for memory savings and inference acceleration with `vllm`.

> `fp8` computation is supported on NVIDIA GPUs with compute capability > 8.9 (Ada Lovelace, Hopper).

## Installation

To get started, install llmcompressor from source as this feature is new:

```bash
pip install git+https://github.com/vllm-project/llm-compressor.git@cb98f34d4ec9dd175e6995d12fb02dec39c6f27a
```

## Quickstart

The example includes an end-to-end script for applying the quantization algorithm:

```bash
python3 llama3_fp8_kv_example.py
```

The resulting model `Meta-Llama-3-8B-Instruct-FP8-KV` is ready to be loaded into vLLM.

## Code Walkthrough

Let's walk through the main steps of the quantization process:

1. Load model
2. Prepare calibration data
3. Apply quantization
4. Evaluate and save the model

### 1. Load Model

Load the model using `AutoModelForCausalLM`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Prepare Calibration Data

Prepare the calibration data using the `ultrachat` dataset:

```python
from datasets import load_dataset

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(text, padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)

ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)
```

### 3. Apply Quantization

Configure and apply the FP8 quantization for weights, activations, and KV cache.
Notice the new `kv_cache_scheme` section:

```python
from llmcompressor import oneshot

recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
```

### 4. Evaluate and Save the Model

Test the quantized model with a sample generation:

```python
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

Save the quantized model:

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-KV"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

For running the model in vLLM, make sure to specify the `kv_cache_dtype="fp8"` argument to enable quantization of the kv cache, and thus usage of your calibrated scales.

## Evaluating Accuracy

To evaluate the accuracy of your quantized model:

1. Install `vllm` and `lm-evaluation-harness`:

```bash
pip install "vllm>=0.5.5" lm_eval==0.4.3
```

2. Run an evaluation (e.g., on GSM-8K):

```bash
MODEL=$PWD/Meta-Llama-3-8B-Instruct-FP8-KV
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,kv_cache_dtype=fp8,add_bos_token=True \
  --tasks gsm8k --num_fewshot 5 --batch_size auto
```

```
vllm (pretrained=Meta-Llama-3-8B-Instruct-FP8-KV,kv_cache_dtype=fp8,add_bos_token=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7748|±  |0.0115|
|     |       |strict-match    |     5|exact_match|↑  |0.7763|±  |0.0115|
```

Note: Include `add_bos_token=True` as quantized models can be sensitive to the presence of the `bos` token.

## Questions or Feature Requests?

Please open an issue on `vllm-project/llm-compressor`.