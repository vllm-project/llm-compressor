# LLM Compressor Entrypoints

## Oneshot

Model optimizations compresses models while preserving accuracy. Oneshot in llm-compressor supports faster infernce on vLLM by applying post-training quantization (PTQ) or sparsification.



### PTQ
PTQ is carried out to reduce the precision of quantizable weights, (ex. Linear layers) to a lower bit-width. Supported formats are 

#### W4A16
- Uses GPTQ for weights to compress weights to 4 bytes. 
- Useful speed ups in low QPS regimes with more weight compression. 
- Recommended for general commodity / enterprize GPUs (NVIDIA A10). 
#### W8A8-int8 
- Uses channel-wise quantization to compress weights to 8 bytes, and uses dynamic per-token quantization to compress activations to 8 bytes. 
- Useful for speed ups in high QPS regimes or offline servering on vLLM. 
- Recommended to be used with NVIDIA A100 GPUs. 
#### W8A8-fp8
- Uses channel-wise quantization to compress weights to 8 bytes, and uses dynamic per-token quantization to compress activations to 8 bytes. 
- Useful for speed ups in high QPS regimes or offline servering on vLLM. 
- Recommended to be used with NVIDIA H100 GPUs. 

### Sparsification
Sparsification reduces model complexity by pruning select weight values to zero while retaining essential weights in a subset of parameters. Supported formats include:
* 2:4
    - Uses what algo?
    - Useful for what case?
    - Recommended hardware?

## Code

Oneshot can be carried out using python or cli. 


### Python 

This is the recommended flow. The example scripts for all the above formats are located in the [examples](../../../examples/) folder. A [W8A8-fp8](../../../examples/quantization_w8a8_fp8/llama3_example.py) example is shown below: 

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


### CLI

Command line argumnets can be carried out under an environment with `llm-compressor` in an editable mode. To install it in editable mode, run:

```bash
pip install -e .
```

Once installed, you can run the oneshot command as follows:

```bash
oneshot --dataset gsm8k --model Xenova/llama2.c-stories15M --output_dir ./oneshot_output --recipe tests/llmcompressor/transformers/oneshot/oneshot_configs/recipes/recipe.yaml --num_calibration_samples 10 --pad_to_max_length False --dataset_config_name main
```




