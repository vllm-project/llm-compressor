from llmcompressor import model_free_ptq

MODEL_ID = "Qwen/Qwen3-0.6B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

# Apply FP8-Block to the model
# Once quantized, the model is saved
# using compressed-tensors to the SAVE_DIR.
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=[
        "model.embed_tokens",
        "lm_head",
    ],
    max_workers=15,
    device="cuda:0",
)
