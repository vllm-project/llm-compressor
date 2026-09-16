## GLM-5.3-Flash NVFP4 Example

GLM-5.3-Flash is a vision-language MoE model whose original [checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash) ships in an FP8-block quantized format. To produce an NVFP4 checkpoint we must first dequantize the model back to full precision (bfloat16), then quantize the MoE expert weights to NVFP4. This walkthrough covers both steps:

1. Decompress the model to bfloat16
2. Apply NVFP4 quantization

### 1. Decompress the Model

The original checkpoint is quantized with FP8-block, so we first convert it to a dense bfloat16 checkpoint using `convert_checkpoint` with the `FP8BlockDequantizer`. The full script can be found in the examples [here](../../../examples/model_free_ptq/decompress_glm53_flash.py).

```python
from compressed_tensors.entrypoints.convert import (
    FP8BlockDequantizer,
    convert_checkpoint,
)

MODEL_ID = "zai-org/GLM-5.3-Flash"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-BF16"

convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=FP8BlockDequantizer.from_pretrained(MODEL_ID),
    max_workers=4,
)
```

This produces a `GLM-5.3-Flash-BF16` checkpoint locally. If you'd rather skip this step, the same bfloat16 checkpoint is published as [`RedHatAI/GLM-5.3-Flash-BF16`](https://huggingface.co/RedHatAI/GLM-5.3-Flash-BF16) and can be used directly as the `MODEL_ID` in the next step.

### 2. Apply NVFP4 Quantization

Once you have the bfloat16 checkpoint (either the local `GLM-5.3-Flash-BF16` directory or `RedHatAI/GLM-5.3-Flash-BF16`), quantize the MoE expert weights to NVFP4 via oneshot. NVFP4 uses static activation quantization, so a calibration dataset is required. Because the model is large, we run calibration distributed via DDP with CPU offloading. The full script can be found in the examples [here](../../../examples/quantizing_moe/glm53_flash_example.py).

Launch with `torchrun`, e.g. `torchrun --nproc-per-node N glm53_flash_example.py`.

```python
import torch
from compressed_tensors.offload import init_dist
from transformers import AutoTokenizer, Glm5NextForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# torchrun --nproc-per-node N ...
init_dist()

# Load the model. Use "RedHatAI/GLM-5.3-Flash-BF16" or the local checkpoint
# produced by decompress_glm53_flash.py.
MODEL_ID = "RedHatAI/GLM-5.3-Flash-BF16"
with load_context(Glm5NextForConditionalGeneration):
    # GLM-5.3-Flash is a vision-language MoE model, so it must be loaded with its
    # `Glm5NextForConditionalGeneration` class (not `AutoModelForCausalLM`) in order
    # to keep the vision tower.
    model = Glm5NextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
recipe = QuantizationModifier(
    scheme="NVFP4",
    targets=[r"re:.*mlp\.experts\..*(gate|up|down)_proj$"],
    ignore=[
        r"re:.*visual.*",  # vision tower stays full precision
        "lm_head",
        r"re:.*mlp\.gate$",  # MoE router
        r"re:.*self_attn\.indexer\..*",  # sensitive to quantization
    ],
)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    recipe=recipe,
    dataset="perfectblend",
    splits="train[:512]",
    max_seq_length=2048,
    num_calibration_samples=512,
)

# Save to disk compressed. MTP tensors (not built by transformers) are copied
# over automatically by the save utility.
model.generation_config.top_p = None
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
```

We quantize only the MoE expert weights (the `gate`/`up`/`down` projections) to NVFP4. The vision tower, `lm_head`, the MoE router (`mlp.gate`), and the attention indexer projections are kept at full precision because they are sensitive to quantization.
