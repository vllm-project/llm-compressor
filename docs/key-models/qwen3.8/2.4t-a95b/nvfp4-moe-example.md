## Qwen3.8-2.4T-A95B NVFP4 + FP8 Example

This example quantizes the Qwen3.8-2.4T-A95B sparse MoE model to a mixed-precision format: NVFP4 for the MoE expert weights and FP8-Block for attention, Gated DeltaNet, and shared expert projections. Because the model is very large, quantization is run with DDP and CPU offloading.

The full example script can be found [here](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantizing_moe/qwen_3_8_example.py).

### Code Walkthrough

1. Load model with DDP
2. Configure quantization recipe
3. Apply quantization
4. Save to disk

### 1. Load Model with DDP

Launch the script with `torchrun`, for example `torchrun --nproc-per-node N qwen_3_8_example.py`. `init_dist()` initializes distributed quantization, and `device_map="auto_offload"` streams weights from CPU as they are needed.

```python
from compressed_tensors.offload import init_dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils import load_context

# torchrun --nproc-per-node N ...
init_dist()

MODEL_ID = "Qwen/Qwen3.8-2.4T-A95B"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={},
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Configure Quantization Recipe

We quantize:

- **Attention and Gated DeltaNet projections** (`self_attn.*` and `linear_attn.*`) to FP8-Block
- **MoE expert weights** (`mlp.*`) to NVFP4
- `lm_head`, the MoE router (`mlp.gate`), and shared expert gates are kept at full precision

```python
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = [
    QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\..*",
                    r"re:.*linear_attn.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$",
                ],
                **FP8_BLOCK,
            ),
            "mlp": QuantizationScheme(
                targets=[r"re:.*mlp\..*"],
                **NVFP4,
            ),
        },
        ignore=[
            "re:.*lm_head",
            "re:.*mlp.gate$",
            "re:.*shared_expert_gate.*",
        ],
    ),
]
```

### 3. Apply Quantization

`pipeline="sequential"` processes the model one layer at a time, keeping memory usage manageable. The prebaked `perfectblend` dataset is used for calibration.

```python
from llmcompressor import oneshot

oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=1024,
    pipeline="sequential",
)
```

### 4. Save to Disk

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```
