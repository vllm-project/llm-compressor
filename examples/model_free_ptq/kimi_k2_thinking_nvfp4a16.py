"""
NOTE: Please run the following script before using `model_free_ptq`

This script is used to reindex the safetensors files of a model such that all fused
modules (gate_up, qkv) are in the same safetensors file. This is required by
model_free_ptq for microscale schemes (NVFP4A16, MXFP4A16)

llmcompressor.reindex_fused_weights \
    unsloth/Kimi-K2-Thinking-BF16 \
    Kimi-K2-Thinking-BF16-reindexed \
    --num_workers=10
"""

from llmcompressor import model_free_ptq

MODEL_ID = "unsloth/Kimi-K2-Thinking-BF16"
REINDEX_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-reindexed"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16"

# See above notice pertaining to safetensors reindexing
# After running `llmcompressor.reindex_fused_weights`,
# use `model_free_ptq` to apply NVFP4A16 quantization
model_free_ptq(
    model_stub=REINDEX_DIR,
    save_directory=SAVE_DIR,
    scheme="NVFP4A16",
    ignore=[
        "re:.*gate$",
        "lm_head",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "model.embed_tokens",
    ],
    max_workers=15,
    device="cuda:0",
)
