---
weight: -8
---

# Compress Your Model

LLM Compressor provides a straightforward way to compress your models using various optimization techniques. This guide walks you through the process of compressing a model with different quantization methods.

## Prerequisites

Before you begin, ensure that your environment meets the following prerequisites:
- **Operating System:** Linux (recommended for GPU support)
- **Python Version:** 3.9 or newer
- **Available GPU:** For optimal performance, it's recommended to use a GPU. LLM Compressor supports the latest PyTorch and CUDA versions for compatibility with NVIDIA GPUs.

## Select a Model and Dataset

Before you start compressing, select the model you'd like to compress and a calibration dataset that is representative of your use case. LLM Compressor supports a variety of models and integrates natively with Hugging Face Transformers and Model Hub, so a great starting point is to use a model from the Hugging Face Model Hub. LLM Compressor also supports many datasets from the Hugging Face Datasets library, making it easy to find a suitable dataset for calibration.

For this guide, we'll use the `TinyLlama` model and the `open_platypus` dataset for calibration. You can replace these with your own model and dataset as needed.

## Select a Quantization Method and Scheme

LLM Compressor supports several quantization methods and schemes, each with its own strengths and weaknesses. The choice of method and scheme will depend on your specific use case, hardware capabilities, and chosen trade-offs between model size, speed, and accuracy.

Supported compression schemes include quantization into W4A16, W8A8‑INT8, and W8A8‑FP8 formats, and sparsification. For a more detailed overview of available quantization schemes, see [Compression Schemes](../guides/compression_schemes.md).

Compression schemes use quantization methods including the following:

| Method | Description | Accuracy Recovery vs. Time |
|--------|-------------|----------------------------|
| **GPTQ** | Utilizes second-order layer-wise optimizations to prioritize important weights/activations and enables updates to remaining weights | High accuracy recovery but more expensive/slower to run |
| **AWQ** | Uses channelwise scaling to better preserve important outliers in weights and activations | Better accuracy recovery with faster runtime than GPTQ |
| **SmoothQuant** | Smooths outliers in activations by folding them into weights, ensuring better accuracy for weight and activation quantized models | Good accuracy recovery with minimal calibration time; composable with other methods |
| **Round-To-Nearest (RTN)** | Simple quantization technique that rounds each value to the nearest representable level in the target precision. | Provides moderate accuracy recovery in most scenarios. Computationally cheap and fast to implement, making it suitable for real-time or resource-constrained environments. |

For this guide, we'll use `GPTQ` composed with `SmoothQuant` to create an `INT W8A8` quantized model. This combination provides a good balance for performance, accuracy, and compatability across a wide range of hardware.

## Apply the Recipe

LLM Compressor provides the `oneshot` API for simple and straightforward model compression. This API allows you to apply a predefined recipe to your model and dataset, making it easy to get started with compression. To apply what we discussed above, we'll import the necessary modifiers and create a recipe to apply to our model and dataset:

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

When you run the above code, the compressed model is saved to the specified output directory: `TinyLlama-1.1B-Chat-v1.0-INT8`. You can then load this model using the Hugging Face Transformers library or vLLM for inference and testing. 

## Memory requirements for LLM Compressor

When compressing a model you should be aware that the memory requirements are dependent on model size and the algorithm used, such as GPTQ/SparseGPT.  

This section will go through how to calculate the CPU and GPU memory requirements for each algorithm using several popular models, an 8B, a 684B, and a model with vision capabilities, as examples. 

The GPTQ/SparseGPT requires a large amount of auxiliary memory. GPTQ/SparseGPT allocates an auxiliary hessian matrix for any layers that are onloaded to the GPU. This is because the hessian matrices have to be almost as large as the weights they are trying to represent. 

Also, larger models, like DeepSeek R1 use a large amount of CPU memory, and models with large vision towers, such as command A, may use large amounts of GPU memory. 

### Things to note when calculating memory requirements for LLM Compressor:

1. A 1B model uses 2Gb of memory to load:
    ```
	mem(1B parameters) ~= (1B parameters) * (2 bytes / parameter) = 2B bytes ~= 2Gb
    ```

2. How text decoder layers and vision tower layers are loaded on to GPU differs significantly. 
    
    In the case of text decoder layers, LLM Compressor dynamically loads one layer at a time into the GPU for computation. The rest of the model remains in CPU memory. 

    However, vision tower layers are loaded onto GPU all at once. Unlike the text model, vision towers are not split up into individual layers before onloading to the GPU. This can create a GPU memory bottleneck for models whose vision towers are larger than their text layers.		

    At this time LLM Compressor does not quantise the vision tower as quantization is generally not worth the tradeoff between latency/throughput and accuracy loss.   

3. LLM Compressor does not currently support tensor parallelism for compression. Supporting this feature will allow layers to be sharded across GPUs, leading to reduced memory usage per GPU and faster compression.

### QuantizationModifier or Round-To-Nearest (RTN)

The quantization modifier, RTN, does not require any additional memory beyond the storage needed for its quantization parameters (scales/zeros). 

If we ignore these scales and zero points from our calculation, we can estimate the following memory requirements:


| Model| CPU requirements | GPU requirements |
|--------|-------------|----------------------------|
| **Meta-Llama-3-8B-Instruct** | mem(8B params) ~= 16Gb | mem(1 Layer) ~= 0.5Gb |
| **DeepSeek-R1-0528-BF16** | mem(684B params) ~= 1368Gb | mem(1 Layer) ~= 22.4Gb|
| **Qwen2.5-VL-7B-Instruct** | mem(7B params) ~= 14Gb | max(mem(1 Text Layer)~= 0.4B, mem(Vision tower)~=1.3B) ~= 1.3Gb |

### GPT Quantization(GPTQ)/ Sparse GPT 

The GPTQ/ SparseGPT algorithms differ from the RTN in that they must also allocate an auxiliary hessian matrices for any layers that are onloaded to the GPU. 

This hessian matrix is used to increase the accuracy recovery of the algorithm, and is approximately the same size as the original weights.

| Model| CPU requirements | GPU requirements |
|--------|-------------|----------------------------|
| **Meta-Llama-3-8B-Instruct** |mem(8B params) ~= 16Gb | mem(1 Layer) * 2 ~= 1Gb |
| **DeepSeek-R1-0528-BF16** | mem(684B params) ~= 1368Gb | mem(1 Layer) * 2 ~= 44.8Gb |
| **Qwen2.5-VL-7B-Instruct** | mem(7B params) ~= 14Gb | max(mem(1 Text Layer)~= 0.4B, mem(Vision tower)~=1.3B)*2 ~= 2.6Gb |


