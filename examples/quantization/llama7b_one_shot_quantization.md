# Creating a Quantized Llama Model in One Shot

Quantizing a model to a lower precision can save on both memory and speed at inference time.
This example demonstrates how to use the SparseML API to quantize a Llama model from 16 bits
to 4 bits and save it to a compressed-tensors format for inference with vLLM.

## Step 1: Select a model and dataset
For this example, we will use a TinyLlama model and the open platypus dataset, however
these can be swapped out for any huggingface compatible models and datasets

```python
model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset = "open_platypus"
```

## Step 2: Configure a `GPTQModifier`
Modifiers in LLMCompressor are used to apply optimizations to models. In this example we use a
`GPTQModifier` to apply the GPTQ algorithm to our model.  We target all `Linear` layers
for 4-bit weight quantization.  These options may be swapped out for any valid `QuantizationScheme`.

```python
from llmcompressor.modifiers.quantization.gptq import GPTQModifier

gptq = GPTQModifier(
    targets="Linear",
    scheme="W4A16"
)
```


### Step3: One-Shot Compression

The `oneshot` api applies the created modifier to the target model and dataset.
Setting `save_compressed` to True runs the model through `compressed_tensors` compression
after the quantization is completed.

```python
from llmcompressor.transformers import oneshot

oneshot(
    model=model,
    dataset=dataset,
    recipe=gptq,
    save_compressed=True,
    output_dir="llama-compressed-example",
    overwrite_output_dir=True,
    max_seq_length=256,
    num_calibration_samples=256,
)
```
