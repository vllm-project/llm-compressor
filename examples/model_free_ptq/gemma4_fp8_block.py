from llmcompressor import model_free_ptq

# Checkpoint available at https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-block

MODEL_ID = "google/gemma-4-31B-it"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8_BLOCK"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=["re:.*vision.*", "lm_head", "re:.*embed_tokens.*"],
    max_workers=8,
    device="cuda:0",
)
