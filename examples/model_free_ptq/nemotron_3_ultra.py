from llmcompressor import model_free_ptq

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

ignore = [
    r"re:.*conv1d.*",
    r"backbone\.embeddings",
    r"re:.*_latent_proj.*",  # sensitive to quantization
    r"re:.*mixer.gate\..*",
    r"re:mtp.layers.*",
    "backbone.norm_f",
    "lm_head",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=ignore,
    max_workers=13,
    device="cuda:0",
)
