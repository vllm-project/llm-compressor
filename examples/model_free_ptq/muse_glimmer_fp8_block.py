from llmcompressor import model_free_ptq

MODEL_ID = "meta-models/Muse-Glimmer-30B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=["re:.*vision.*", "lm_head", "re:.*embed_tokens.*"],
    max_workers=15,
    device="cuda:0",
)
