# Applying 2:4 Sparsity with Optional FP8 Quantization

This script demonstrates how to apply **2:4 structured sparsity** with and without **FP8 quantization** to the `Meta-Llama-3-8B-Instruct` model using the `llm-compressor` library. The compressed model is optimized for memory efficiency and faster inference on supported GPUs.

> **Note:** FP8 dynamic precision computation is supported on Nvidia GPUs with CUDA Compute Capability 9.0 and above.


## Installation

To get started, install the `llm-compressor` library and its dependencies:

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## Quickstart
Run the script with the following commands:

- Without FP8 Quantization:

```bash
python3 llama3_8b_2of4.py
```

- With FP8 Quantization:

```bash
python3 llama3_8b_2of4.py --fp8
```

The script compresses the Meta-Llama-3-8B-Instruct model using:

- **2:4 Structured Sparsity:** Applies structured pruning to reduce weights by 50%.
- **FP8 Quantization (Optional):** Enables dynamic quantization for further memory savings.


### Configuration

- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Dataset: HuggingFaceH4/ultrachat_200k (train split)
- Calibration Samples: 512
- Maximum Sequence Length: 2048

## Steps to Run

1. **Select Model, Dataset, and Recipe**

The model and dataset are predefined in the script. The recipe dynamically adjusts the recipe based on 
whether FP8 quantization is enabled (--fp8 flag).

Example Recipe:

```python
recipe = [
    SparseGPTModifier(
        sparsity=0.5,
        mask_structure="2:4",
        sequential_update=True,
        targets=[r"re:model.layers.\d*$"],
    )
]

if fp8_enabled:
    recipe.append(
        QuantizationModifier(
            targets=["Linear"],
            ignore=["lm_head"],
            scheme="FP8_DYNAMIC",
        ),
    )
```

2. **Apply Compression**
The script applies compression using the oneshot function:

```python
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

### Saving the Compressed Model

The compressed model and tokenizer are saved to the output directory:

```python
model.save_pretrained(save_dir, save_compressed=True)
tokenizer.save_pretrained(save_dir)
```

Output Directories:
- Without FP8: `Meta-Llama-3-8B-Instruct-2of4-sparse`
- With FP8: `Meta-Llama-3-8B-Instruct-2of4-W8A8-FP8-Dynamic-Per-Token`

#### Saving Without Sparse Compression

To save the model on disk without sparse compression:

```python
model.save_pretrained(save_dir, save_compressed=True, disable_sparse_compression=True)
tokenizer.save_pretrained(save_dir)
```

> **Note:** Saving a model with both the `save_compressed` and `disable_sparse_compression` options will compress the model using the quantization compressor; however, instead of using the more disk-efficient sparsity compressor(s), the dense sparsity compressor will be used. The `dense` sparsity compressor saves model params as is, and does not leverage sparsity for disk-efficient storage. These options only affect how the model(s) are saved on disk and do not impact the actual pruning or quantization processes.

### Validation

After compression, the script validates the model by generating a sample output:

```plaintext
========== SAMPLE GENERATION ============
Hello my name is ...
=========================================
```

**Notes:** 
- Ensure your GPU has sufficient memory (at least ~25GB) to run compression script.
- Use the `--fp8` flag to enable FP8 quantization.

Modify the `MODEL_ID` and `DATASET_ID` variables to use other models or datasets.
### Running in vLLM

Install vLLM using `pip install vllm`

```python

# run_model.py

import argparse
from vllm import LLM, SamplingParams

def run_inference(model_path, tensor_parallel_size, prompt="Hello my name is:"):
    """
    Loads a model and performs inference using LLM.
    """
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
    )
    # Load the model
    model = LLM(
        model=model_path, 
        enforce_eager=True,
        dtype="auto",
        tensor_parallel_size=tensor_parallel_size,
    )

    # Generate inference
    outputs = model.generate(prompt, sampling_params=sampling_params)
    return outputs[0].outputs[0].text

def main():
    """ Main function to handle CLI and process the model. """
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run inference on a single model and print results.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model to perform inference."
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for the model. Default is 1."
    )

    args = parser.parse_args()
    model_path = args.model_path
    tensor_parallel_size = args.tensor_parallel_size

    prompt = "Hello my name is:"

    # Load model and perform inference
    inference_result = run_inference(model_path, tensor_parallel_size)
    print("="* 20)
    print("Model:", model_path)
    print(prompt, inference_result)

if __name__ == "__main__":
    main()
```

Command to run model:
```bash
python3 run_model.py <MODEL_PATH>
```

Example:
```bash
python3 run_model.py Meta-Llama-3-8B-Instruct2of4-W8A8-FP8-Dynamic-Per-Token
```

