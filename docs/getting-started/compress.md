---
weight: -8
---

# Compress Your Model

LLM Compressor provides a straightforward way to compress your models using various optimization techniques. This guide will walk you through the process of compressing a model using different quantization methods.

## Prerequisites

Before you begin, ensure you have the following prerequisites:
- **Operating System:** Linux (recommended for GPU support)
- **Python Version:** 3.9 or newer
- **Available GPU:** For optimal performance, it's recommended to use a GPU. LLM Compressor supports the latest PyTorch and CUDA versions for compatability with NVIDIA GPUs.

## Select a Model and Dataset

Before you start compressing, select the model you'd like to compress and a calibration dataset that is representative of your use case. LLM Compressor supports a variety of models and integrates natively with Hugging Face Transformers and Model Hub, so a great starting point is to use a model from the Hugging Face Model Hub. LLM Compressor also supports many datasets from the Hugging Face Datasets library, making it easy to find a suitable dataset for calibration.

For this guide, we'll use the `TinyLlama` model and the `open_platypus` dataset for calibration. You can replace these with your own model and dataset as needed.

## Select a Quantization Method and Scheme

LLM Compressor supports several quantization methods and schemes, each with its own strengths and weaknesses. The choice of method and scheme will depend on your specific use case, hardware capabilities, and desired trade-offs between model size, speed, and accuracy.

Some common quantization schemes include:

| Scheme | Description | Hardware Compatibility |
|--------|-------------|------------------------|
| **FP W8A8** | 8-bit floating point (FP8) quantization for weights and activations, providing ~2X smaller weights with 8-bit arithmetic operations. Good for general performance and compression, especially for server and batch inference. | Latest NVIDIA GPUs (Ada Lovelace, Hopper, and later) and latest AMD GPUs |
| **INT W8A8** | 8-bit integer (INT8) quantization for weights and activations, providing ~2X smaller weights with 8-bit arithmetic operations. Good for general performance and compression, especially for server and batch inference. | All NVIDIA GPUs, AMD GPUs, TPUs, CPUs, and other accelerators |
| **W4A16** | 4-bit integer (INT4) weights with 16-bit floating point (FP16) activations, providing ~3.7X smaller weights but requiring 16-bit arithmetic operations. Maximum compression for latency-sensitive applications with limited memory. | All NVIDIA GPUs, AMD GPUs, TPUs, CPUs, and other accelerators |

Some common quantization methods include:

| Method | Description | Accuracy Recovery vs. Time |
|--------|-------------|----------------------------|
| **GPTQ** | Utilizes second-order layer-wise optimizations to prioritize important weights/activations and enables updates to remaining weights | High accuracy recovery but more expensive/slower to run |
| **AWQ** | Uses channelwise scaling to better preserve important outliers in weights and activations | Moderate accuracy recovery with faster runtime than GPTQ |
| **SmoothQuant** | Smooths outliers in activations by folding them into weights, ensuring better accuracy for weight and activation quantized models | Good accuracy recovery with minimal calibration time; composable with other methods |

For this guide, we'll use `GPTQ` composed with `SmoothQuant` to create an `INT W8A8` quantized model. This combination provides a good balance for performance, accuracy, and compatability across a wide range of hardware.

## Apply the Recipe

LLM Compressor provides the `oneshot` API for simple and straightforward model compression. This API allows you to apply a pre-defined recipe to your model and dataset, making it easy to get started with compression. To apply what we discussed above, we'll import the necessary modifiers and create a recipe to apply to our model and dataset:

```python
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]
oneshot(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="TinyLlama-1.1B-Chat-v1.0-INT8",
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

Once the above code is run, it will save the compressed model to the specified output directory: `TinyLlama-1.1B-Chat-v1.0-INT8`. You can then load this model using the Hugging Face Transformers library or vLLM for inference and testing.
