# Compress Your Model

LLM Compressor provides a straightforward way to compress your models using various optimization techniques. This guide walks you through the process of compressing a model with different quantization methods.

## Prerequisites

Before you begin, ensure that your environment meets the following prerequisites:
- **Operating System:** Linux (recommended for GPU support)
- **Python Version:** 3.10 or newer
- **Available GPU:** For optimal performance, it's recommended to use a GPU. LLM Compressor supports the latest PyTorch and CUDA versions for compatibility with NVIDIA GPUs.

## Compress your model through oneshot

LLM Compressor provides the `oneshot` API for simple and straightforward model compression. This API allows you to apply a recipe, which defines your chosen quantization scheme and quantization algorithm, to your selected model. 
We'll import the `QuantizationModifier` modifier, which applies the RTN quantization algorithm and create a recipe to apply FP8 Block quantization to our model. The final model is compressed in the compressed-tensors format and ready to deploy in vLLM.

!!! info
    Note: The following script is for single-process quantization. The model is loaded onto any available GPUs and then offloaded onto the cpu if it is too large. For distributed support or support for very large models (such as certain MoEs, including Kimi-K2), see the [Big Models and Distributed Support guide](../guides/big_models_and_distributed/model_loading.md).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.offload import dispatch_model

MODEL_ID = "Qwen/Qwen3-30B-A3B"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to FP8 using RTN with block_size 128
#   * quantize the activations dynamically to FP8 during inference
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head", "re:.*mlp.gate$"],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-BLOCK"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

When you run the above code, the compressed model is saved to the specified output directory: `Qwen3-30B-A3B-FP8-BLOCK`. You can then load this model using the Hugging Face Transformers library or vLLM for inference and testing. 

## Next steps

- [Deploy with vLLM](deploy.md)