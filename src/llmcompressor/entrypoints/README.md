# LLM Compressor Entrypoints

## Oneshot

Model optimizations compress models while preserving accuracy. One-shot in LLM-Compressor supports faster inference on vLLM by applying post-training quantization (PTQ) or sparsification

### PTQ
PTQ is performed to reduce the precision of quantizable weights (e.g., linear layers) to a lower bit-width. Supported formats are:

#### W4A16
- Uses GPTQ to compress weights to 4 bits. 
- Useful speed ups in low QPS regimes with more weight compression. 
- Recommended for any GPUs types. 
#### W8A8-INT8 
- Uses channel-wise quantization to compress weights to 8 bits, and uses dynamic per-token quantization to compress activations to 8 bits. 
- Useful for speed ups in high QPS regimes or offline serving on vLLM. 
- Recommended for NVIDIA GPUs with compute capability <8.9 (Ampere, Turing, Volta, Pascal, or older). 
#### W8A8-FP8
- Uses channel-wise quantization to compress weights to 8 bits, and uses dynamic per-token quantization to compress activations to 8 bits. 
- Useful for speed ups in high QPS regimes or offline serving on vLLM. 
- Recommended for NVIDIA GPUs with compute capability >8.9 (Hopper and Ada Lovelace). 

### Sparsification
Sparsification reduces model complexity by pruning selected weight values to zero while retaining essential weights in a subset of parameters. Supported formats include:

#### 2:4-Sparsity
- Uses semi-structured sparsity, where for every four contiguous weights in a tensor, two are set to zero. 
- Useful for efficiently computing sparse matrix multiplications using NVIDIA Sparse Tensor Cores. To preserve as much accuracy from the base model, usually a sparse-finetune step is added. 
- Recommended for NVIDIA architecture from Amphere onwards, supported on GPUs with Sparse Tensor Cores

## Code

Example scripts for all the above formats are located in the [examples](../../../examples/) folder. A [W8A8-FP8](../../../examples/quantization_w8a8_fp8/llama3_example.py) example is shown below: 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

oneshot(model=model, recipe=recipe)
```
