## Qwen3.8-2.4T-A95B REAP + NVFP4 Example

[REAP](https://arxiv.org/pdf/2510.13999) prunes the least-salient experts from a sparse MoE model before quantization. This example prunes 25% of the experts in each MoE layer of Qwen3.8-2.4T-A95B, then quantizes the remaining model to NVFP4, reducing VRAM requirements while preserving accuracy. The published `REAP-25` and `REAP-50` checkpoints use 25% and 50% expert sparsity respectively.

The full example script can be found [here](https://github.com/vllm-project/llm-compressor/blob/main/examples/reap_expert_pruning/qwen38_example.py).

### Code Walkthrough

1. Load model with DDP
2. Configure pruning and quantization
3. Apply the recipe
4. Save to disk

### 1. Load Model with DDP

Launch the script with `torchrun`, for example `torchrun --nproc-per-node N qwen38_example.py`. The model is streamed from CPU with `device_map="auto_offload"`.

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
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Configure Pruning and Quantization

The `REAPPruningModifier` removes the least-salient experts in each MoE layer, and the `QuantizationModifier` then quantizes the pruned model. Set `sparsity=0.5` to reproduce the `REAP-50` checkpoint.

```python
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = [
    REAPPruningModifier(sparsity=0.25),
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

### 3. Apply the Recipe

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
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8-REAP-25"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```
