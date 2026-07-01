from llmcompressor import model_free_ptq

MODEL_ID = "<MODEL_ID>"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-<SCHEME>"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="<SCHEME>",  # FP8_DYNAMIC | FP8_BLOCK
    ignore=[
        "lm_head",
    ],
    max_workers=15,
    device="cuda:0",
)
