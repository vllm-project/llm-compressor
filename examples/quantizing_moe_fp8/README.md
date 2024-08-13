# Quantizing TinyMixtral-4x248M-MoE Model with FP8

This directory contains an example script for quantizing the `TinyMixtral-4x248M-MoE` model using the FP8 quantization scheme.

## Installation

To get started, install the necessary dependencies by executing the following commands:

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## Quickstart

The provided example script demonstrates an end-to-end process for applying the quantization algorithm:

```bash
python3 tinymixtral_4_248_moe.py
```

## Creating a Quantized MoE Model

This example leverages `LLMCompressor` and `Compressed-Tensors` to create an FP8-quantized `TinyMixtral 4x248M MoE` model. The model is calibrated and trained using the `open_platypus` dataset.

You can follow the detailed steps below or simply run the example script with:

```bash
python examples/quantizing_moe_fp8/tinymixtral_4_248_moe.py
```

### Step 1: Select a Model, Dataset, and Recipe

In this step, you'll choose a baseline model for quantization, a dataset for calibration, and a quantization recipe.

- **Models**: Can be referenced from a local directory or retrieved from the Hugging Face Hub.
- **Datasets**: Can also be from a local directory or the Hugging Face Hub.
- **Recipes**: These are YAML files or Python modifier objects that describe how a model should be optimized during or after training. In this example, we use a `GPTQModifier` object with the scheme set to `FP8`.

```python
from llmcompressor.modifiers.quantization.gptq import GPTQModifier

recipe = GPTQModifier(scheme="FP8", targets="Linear", ignore=["lm_head"])
```

### Step 2: Run Quantization Using Oneshot

The `oneshot` method applies the selected recipe to your model and dataset without requiring any fine-tuning. The model will be sparsified and saved to `"tinymixtral-4-248-moe_quantized_fp8"`.

```python
from llmcompressor.transformers import oneshot

output_dir = "tinymixtral-4-248-moe_quantized_fp8"

oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    save_compressed=True,
    output_dir=output_dir,
    overwrite_output_dir=True,
    max_seq_length=128,
    num_calibration_samples=1000,
)
```

### Custom Quantization

The repository supports multiple quantization techniques configured via a recipe. Supported strategies include `tensor`, `group`, and `channel` quantization.

In the above example, FP8 channel-wise quantization is used as specified by the `FP8` scheme. For other preset schemes, refer to the [quantization schemes](https://github.com/neuralmagic/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py) in the `Compressed-Tensors` library.

A custom scheme can also be specified using `config_groups`:

```python
# Example of defining a custom quantization scheme

from llmcompressor.modifiers.quantization.gptq import GPTQModifier

config_groups = {
                "group_0": {
                    "targets": ["Linear"],
                    "input_activations": None,
                    "output_activations": None,
                    "weights": {
                        "num_bits": 8
                        "type": "int"
                        "symmetric": true
                        "strategy": "tensor"
                        "group_size": 128 
                    }
               }
}

recipe = GPTQModifier(config_groups=config_groups)
```