## Kimi-K3 NVFP4 Example

### Overview

Kimi-K3 requires custom modeling files bundled with LLM Compressor, since it is not yet supported in Transformers.
The example below quantizes the model to NVFP4 using calibration data.

This model has already been quantized to mxfp4, so we must first dequantize it before
applying a new quantization

The full example script can be found [here](../../../examples/quantizing_moe/kimi_k3_example.py).

### Code Walkthrough

```python
# requires: einops, fla-core, tiktoken
import torch.distributed as dist
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.distributed import init_dist
from transformers import AutoConfig, AutoProcessor, CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Small representative model with same MXFP4 quantization
MODEL_ID = "inference-optimization/Kimi-K3-0.40B-MXFP4"  # "moonshotai/Kimi-K3"

# Patch quantization config to
# 1. Fix an incomplete ignore list provided by the base checkpoint
# 2. Disable decompression (for later step)
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
qconfig = CompressedTensorsConfig(
    **config.quantization_config, dequantize=False, use_optimized_inference=False
)
qconfig.quantization_config.ignore += [
    "re:.*mlp_res_proj.*",
    "re:.*self_attention_res_proj.*",
    "re:.*routed_expert.*",
    "re:.*output_attn_res_proj.*",
]

# Load model with the modified quantization config and disk offloading
init_dist()
with load_context(KimiK3ForConditionalGeneration):
    model = KimiK3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=qconfig,
        device_map="auto_offload",
        trust_remote_code=True,
        max_memory={},
        offload_folder="offload_folder",
    )
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Decompress model upfront before calibrating
ModelCompressor.from_pretrained_model(model).decompress_model(model)

recipe = QuantizationModifier(
    targets="re:.*mlp.*",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        r"re:.*block_sparse_moe\.gate",
        "re:.*vision_tower.*",
        "re:.*mlp_res_proj$",
    ],
)

oneshot(
    model=model,
    tokenizer=processor.tokenizer,
    dataset="perfectblend",
    splits=get_rank_partition("train", 512),
    recipe=recipe,
    max_seq_length=2048,
    trust_remote_code_model=True,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

dist.destroy_process_group()
```
