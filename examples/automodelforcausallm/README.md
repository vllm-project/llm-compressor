# Loading models using `AutoModelForCausalLM`

Models quantized through `llm-compressor` can be loaded directly through 
`AutoModelForCausalLM`. Note: this requires `transformers>=v4.45.0` and 
`compressed-tensors>v0.6.0`.

```python
from transformers import AutoModelForCausalLM

MODEL_ID = "nm-testing/tinyllama-w8a8-compressed-hf-quantizer"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
```
