from llmcompressor import model_free_ptq

MODEL_ID = "Tesslate/OmniCoder-9B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"

# Apply FP8-Dynamic to the model
# Once quantized, the model is saved
# using compressed-tensors to the SAVE_DIR.
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",
        "re:.*model.embed_tokens.*",
        "re:.*visual.*",
        "re:.*conv1d.*",
    ],
    max_workers=15,
    device="cuda:0",
)
