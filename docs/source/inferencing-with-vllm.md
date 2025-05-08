# Inferencing with vLLM

The checkpoints created by llmcompressor can be loaded and run in vllm.

Install `vllm`:

```cmd
pip install vllm
```

Run:

```py
from vllm import LLM
model = LLM("TinyLlama-1.1B-Chat-v1.0-INT8")
output = model.generate("My name is")
```