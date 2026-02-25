## Mistral Large 3 FP8 Example

### Code Walkthrough

#### Prerequisite: Script

```bash
#NOTE: Please run the following script before using `model_free_ptq`

#

# This script is used to reindex the safetensors files of a model such that all fused
# modules (gate_up, qkv) are in the same safetensors file. This is required by
# model_free_ptq for microscale schemes (NVFP4A16, MXFP4A16)

llmcompressor.reindex_fused_weights \
    mistralai/Mistral-Large-3-675B-Instruct-2512-BF16 \
    Mistral-Large-3-675B-Instruct-2512-BF16-reindexed \
    --num_workers=10
```

Let's walk through the main steps of the quantization process:
1. Load model
2. Apply quantization
3. Modify ignore list

### 1. Load Model

```python
from llmcompressor import model_free_ptq

MODEL_ID = "mistralai/Mistral-Large-3-675B-Instruct-2512-BF16"
REINDEX_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-reindexed"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"
```

### 2. Apply Quantization

```python
model_free_ptq(
    REINDEX_DIR,
    SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=[
        "tok_embeddings",  # embeddings
        "re:patch_merger.*",  # patch merger
        "re:vision_encoder.*",  # vision tower
        "re:vision_language_adapter.*",  # vision adapter
        "re:.*wkv_a_with_mqa$",  # non divisible
        "re:.*wq_a$",  # fused with wkv_a_with_mqa
        "re:.*gate$",  # gate layers
        "output",  # lm head
    ],
    max_workers=10,
    device="cuda:0",
)
```

### 3. Modify Ignore List

vLLM uses different weight names than the names of the huggingface transformers model. To reflect this, please update the ignore list in `params.json` to the following:

```json
 "ignore": [
   "model.embed_tokens",
   "re:patch_merger.*",
   "re:vision_encoder.*",
   "re:vision_language_adapter.*",
   "re:.*kv_a_proj_with_mqa$",
   "re:.*q_a_proj$",
   "re:.*gate$",
   "lm_head"
 ],
```

