## Kimi-K2.6 FP8 Block Example

### Code Walkthrough

The original [Kimi K2.6 checkpoint](https://huggingface.co/moonshotai/Kimi-K2.6) ships in a quantized format with 4-bit integer weights. 
In order to create an FP8 block checkpoint, we must first dequantize to full-precision (bfloat16), then quantize to the desired FP8 Block format. 
Fortunately, this can be done in a single call to the `model_free_ptq` entrypoint because FP8 block quantization does not require a calibration dataset.
The original 4-bit weights will be loaded from the safetensors files, upconverted to bfloat16, and quantized to FP8 block in a single pipeline.

The full example script can be found in the examples [here](../../../../examples/model_free_ptq/kimi_k26_fp8_block.py).

The snippet below was run successfully on a single H100x80GB GPU.

### 1. Convert model from 4-bit integer weights to fp8 block format.

```python
from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer

from llmcompressor import model_free_ptq

MODEL_ID = "moonshotai/Kimi-K2.6"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*kv_a_proj_with_mqa$",
    "re:.*q_a_proj$",
    "re:.*vision_tower.*",
    "re:.*embed_tokens$",
    # ignore anything not in language_model
    "re:.*mm_projector.*",
    "re:.*vision.*",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=ignore,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        ignore=ignore,
    ),
    max_workers=2,
    device="cuda:0",
)
```
