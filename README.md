# LLM Compressor

`llm-compressor` is an easy-to-use library for optimizing models for deployment with `vllm`, including:
* Comprhensive set of quantization algorithms including weight-only and activation quantization
* Seemless integration Hugging Face models and repositories
* `safetensors`-based file format compatible with `vllm`

### Supported Formats
* Mixed Precision: W4A16, W8A16
* Integer Quantization: W8A8 (int8)
* Floating Point Quantization: W8A8 (fp8)
* 2:4 Semi-structured Sparsity
* Unstructured Sparsity

### Supported Algorithms
* PTQ (Post Training Quantization)
* GPTQ
* SmoothQuant
* SparseGPT


## Installation

`llm-compressor` can be installed from the source code via a git clone and local pip install.

```bash
git clone https://github.com/vllm-project/llm-compressor.git
pip install -e llm-compressor
```

## Quick Tour
The following snippet is a minimal example for compression and inference of a `Meta-Llama-3-8B-Instruct`, but the model can be swapped for a local or remote HF-compatible checkpoint.

This example uses 4-bit weight-only quantization, however the `scheme` may be changed to target different quantization algorithms.


### Compression
Compression is easily applied by selecting an algorithm (GPTQ) and calling the `oneshot` API.

```python
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization.gptq import GPTQModifier

# sets parameters for the GPTQ algorithms - target Linear layer weights at 4 bits
gptq = GPTQModifier(scheme="W4A16", targets="Linear")

oneshot(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # sample model
    dataset="open_platypus",  # calibration dataset, swap with yours
    recipe=gptq,
    save_compressed=True,
    output_dir="llama-compressed-quickstart",
    overwrite_output_dir=True,
    max_seq_length=256,
    num_calibration_samples=256,
)
```

### Inference with vLLM
To run inference with vLLM, first install vLLM from pip `pip install vllm`.

```python
from vllm import LLM

model = LLM("llama-compressed-quickstart")
output = model.generate("I love 4 bit models because")
```


## Learn More
The llm-compressor library provides a rich feature-set for model compression below are examples
and documentation of a few key flows.  If you have any questions or requests
open an [issue](https://github.com/vllm-project/llm-compressor/issues) and we will add an example or documentation.

* One-Shot API Documentation (Coming Soon!)
* Modifiers Documentation (Coming Soon!)
* [One-Shot Quantization](examples/quantization/llama7b_one_shot_quantization.md)
* [Creating a Sparse-Quantized Llama-7b model](examples/quantization/llama7b_one_shot_quantization.md)

## Contribute
We appreciate contributions to the code, examples, integrations, and documentation as well as bug reports and feature requests!
[Learn how here](CONTRIBUTING.md).