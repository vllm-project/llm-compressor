from llmcompressor import model_free_ptq

MODEL_ID = "unsloth/Kimi-K2-Thinking-BF16"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16"

# See above notice pertaining to safetensors reindexing
# After running `llmcompressor.reindex_fused_weights`,
# use `model_free_ptq` to apply NVFP4A16 quantization
model_free_ptq(
    model_stub=MODEL_ID,
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
