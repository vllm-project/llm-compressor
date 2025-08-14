# `fp4` Quantization

`llm-compressor` supports quantizing weights and activations to `fp4` for memory savings and inference acceleration with `vLLM`. In particular, `nvfp4` is supported - a 4-bit floating point encoding format introduced with the NVIDIA Blackwell GPU architecture.

## Installation

To get started, install:

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## Quickstart

The example includes an end-to-end script for applying the quantization algorithm.

```bash
python3 llama3_example.py
```

The resulting model `Meta-Llama-3-8B-Instruct-NVFP4` is ready to be loaded into vLLM.
Note: if running inference on a machine that is < SM100, vLLM will not run activation
quantization, only weight-only quantization.

## Code Walkthough

Now, we will step though the code in the example:
1) Load model
2) Prepare calibration data
3) Apply quantization

### 1) Load Model

Load the model using `AutoModelForCausalLM` for handling quantized saving and loading. 

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2) Prepare Calibration Data

Prepare the calibration data. `nvfp4` quantization generates per-tensor global scales and per-group (size 16) local quantization scales for the weights, as well as per-tensor global scales for the activations. Per-group local activation quantization scales are generated dynamically during inference time. We need some sample data to calibrate the global activation scales. Typically, a small number of samples is sufficient. In this example, we use a sample size of 20.

It is useful to use calibration data that closely matches the type of data used in deployment. If you have fine-tuned a model, using a sample of your training data is a good idea. In our case, we are quantizing an instruction-tuned generic model, so we will use the `ultrachat` dataset. 

### 3) Apply Quantization

With the dataset ready, we will now apply quantization.

We first select the quantization algorithm.

In our case, we will apply the default QuantizationModifier recipe for `nvfp4` to all linear layers.
> See the `Recipes` documentation for more information on making complex recipes

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the quantization algorithm to run.
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

We have successfully created an `nvfp4` model!

# Quantizing MoEs

To quantize MoEs, a few additional steps are required. An example quantizing Llama4 can be found under `llama4_example.py`. Here, we replace all `Llama4TextMoe` modules by calling `replace_modules_for_calibration`. This replacement allows us to:

1. Linearize the model to enable quantization and execution in vLLM. This is required as the native model definition does not include `torch.nn.Linear` layers in its MoE blocks, a requirement for LLM Compressor to run quantization.
2. Ensure experts are quantized correctly as not all experts are activated during calibration

Similarly, an example quantizing the Qwen3-30B-A3B model can be found under `qwen_30b_a3b.py`. This model does not require additional linearization as required by the Llama4 model. However, similar to Llama4, in order to ensure the experts are quantized correctly, we can pass in `calibrate_moe_context` which temporarily updates the model definition to use `Qwen3MoeSparseMoeBlock` which updates how the forward pass is handled in the MoE block during calibration. Feel free to update the definition under `llm-compressor/src/llmcompressor/modeling/qwen3_moe.py` to play around with this behavior and evaluate its impact on quantization performance.


