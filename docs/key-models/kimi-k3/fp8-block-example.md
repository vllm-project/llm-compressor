## Kimi-K3 FP8 Block Example

### Overview

This example uses `model_free_ptq` to quantize Kimi-K3 to FP8 block format without loading the full model into memory.
The original checkpoint ships pre-quantized, so a `CompressedTensorsDequantizer` is used to dequantize on the fly during conversion.

The full example script can be found [here](../../../examples/model_free_ptq/kimi_k3_fp8_block.py).

### Code Walkthrough

```python
from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer

from llmcompressor import model_free_ptq

MODEL_ID = "moonshotai/Kimi-K3"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

# no attention because (q_proj|k_proj|v_proj|b_proj|f_a_proj) are all fused
# and `b_proj` has weight shape [96, 7168] which is not divisible by 128
ignore = [
    "re:.*embed_tokens.*",
    "re:.*self_attn.*",
    "re:.*block_sparse_moe\.gate.*",
    "re:.*self_attention_res_proj.*",
    "re:.*mlp_res_proj.*",
    "re:.*output_attn_res_proj.*",
    "re:.*lm_head.*",
    "re:.*vision_tower.*",
    "re:.*mm_projector.*",
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
    max_workers=7,
    device=[f"cuda:{i}" for i in range(7)],
)
```
