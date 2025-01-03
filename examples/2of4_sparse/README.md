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
    recipe.extend([
        QuantizationModifier(
            targets=["Linear"],
            ignore=["lm_head"],
            scheme="FP8_DYNAMIC",
        ),
        ConstantPruningModifier(
            targets=[
                r"re:.*q_proj.weight", r"re:.*k_proj.weight", r"re:.*v_proj.weight",
                r"re:.*o_proj.weight", r"re:.*gate_proj.weight", r"re:.*up_proj.weight",
                r"re:.*down_proj.weight",
            ],
            start=0,
        ),
    ])
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

3. **Save the Compressed Model**

The compressed model and tokenizer are saved to the output directory:

```python
model.save_pretrained(save_dir, save_compressed=True)
tokenizer.save_pretrained(save_dir)
```

Output Directories:
- Without FP8: `Meta-Llama-3-8B-Instruct-2of4-sparse`
- With FP8: `Meta-Llama-3-8B-Instruct-2of4-W8A8-FP8-Dynamic-Per-Token`

### Validation

After compression, the script validates the model by generating a sample output:

```plaintext
========== SAMPLE GENERATION ============
Hello my name is ...
=========================================
```

**Notes:** 
- Ensure your GPU has sufficient memory (at least ~25GB) to run this script.
- Use the `--fp8` flag to enable FP8 quantization.

Modify the `MODEL_ID` and `DATASET_ID` variables to use other models or datasets.





