# Loading models using `AutoModelForCausalLM`

Models quantized through `llm-compressor` can be loaded directly through 
`AutoModelForCausalLM`. Note: this requires `transformers>=v4.45.0` and 
`compressed-tensors>v0.6.0`.

```python
from transformers import AutoModelForCausalLM

MODEL_ID = "nm-testing/tinyllama-w8a8-compressed-hf-quantizer"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
```

These models are still supported through the `SparseAutoModelForCausalLM` pathway:

```python
from llmcompressor.transformers import SparseAutoModelForCausalLM

MODEL_ID = "nm-testing/tinyllama-w8a8-compressed-hf-quantizer"
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
)
```

Models quantized through `llm-compressor` using `compressed-tensors=<v0.6.0` are not
supported through the `AutoModelForCausalLM` and will still need the 
`SparseAutoModelForCausalLM` pathway to run.