# Quantizing Big Models with HF Accelerate

`llmcompressor` integrates with `accelerate` to support quantizing large models such as Llama 70B and 405B.

## Overview

[`accelerate`]((https://huggingface.co/docs/accelerate/en/index)) is a highly useful library in the Hugging Face ecosystem that supports for working with large models, including:
- Offloading parameters to CPU
- Sharding models across multiple GPUs with pipeline-parallelism


### Using `device_map`

To enable `accelerate` features with `llmcompressor`, simple insert `device_map` in `from_pretrained` during model load.

```python
from llmcompressor.transformers import SparseAutoModelForCausalLM
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"

# device_map="auto" triggers usage of accelerate
# if > 1 GPU, the model will be sharded across the GPUs
# if not enough GPU memory to fit the model, parameters are offloaded to the CPU
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto")
```

`llmcompressor` is designed to respect the `device_map`, so calls to `oneshot` will work properly.

### Practical Advice

When working with `accelerate`, it is important to keep in mind that CPU offloading and naive pipeline-parallelism will slow down forward passes through the model. As a result, we need to take care to ensure that the quantization methods used fit well with the offloading scheme as methods that require many forward passes though the model will be slowed down.

General rules of thumb:
- CPU offloading should only be with data-free quantization methods (e.g. PTQ with `FP8_DYNAMIC`)
- Multi-GPU is fast enough to be used with calibration data-based methods with `sequential_update=False`
- It is possible to use Multi-GPU with `sequential_update=True`, but the runtime will be slower

## Examples

We will show working examples for each use case:
- **CPU Offloading**: Quantize Llama-70B to `FP8` using `PTQ` with a single GPU
- **Multi-GPU**: Quantize Llama-70B to `INT8` using `GPTQ` and `SmoothQuant` with 8 GPUs

### Installation

Install `llmcompressor`:

```bash
pip install llmcompressor==0.1.0
```

### CPU Offloading: `FP8` Quantization

CPU offloading is slow. As a result, we recommend using this feature only with data-free quantization methods. For example, when quantizing a model to `fp8`, we typically use simple `PTQ` to statically quantize the weights and use dynamic quantization for the activations. These methods do not require calibration data.

- `cpu_offloading_fp8.py` demonstrates quantizing the weights and activations of `Llama 70B` to `fp8` on a single GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
python cpu_offloading_fp8.py
```

The resulting model `./Meta-Llama-3-70B-Instruct-FP8-Dynamic` is ready to run with `vllm`!

### Multi-GPU: `INT8` Quantization with `GPTQ` and `SmoothQuant`

For quantization methods that require calibration data (e.g. `GPTQ` and `SmoothQuant`), CPU offloading is too slow. For these methods, `llmcompressor` can use `accelerate` multi-GPU to quantize models that are larger than a single GPU. For example, when quantizing a model to `int8`, we typically use `GPTQ` to statically quantize the weights and `SmoothQuant` to make the activations easier to quantize. Each of these methods require calibration data.

- `multi_gpu_int8.py` demonstrates quantizing the weights and activations of `Llama-70B` to `int8` on 8 A100s:

```python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python multi_gpu_int8.py
```

The resulting model `./Meta-Llama-3-70B-Instruct-INT8-Dynamic` is quantized and ready to run with `vllm`!

## Questions or Feature Request?

Please open up an issue on `vllm-project/llm-compressor`