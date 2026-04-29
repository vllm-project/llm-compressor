from llmcompressor import model_free_ptq
from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer

# moonshotai/Kimi-K2.6 checkpoint is published in compressed-tensors format.
# This script will upconvert to bfloat16 so that the model can be compressed
# to FP8_BLOCK

MODEL_ID = "moonshotai/Kimi-K2.6"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*kv_a_proj_with_mqa$",
    "re:.*q_a_proj$",
    "re:.*vision_tower.*",
    "re:.*embed_tokens$",
    "re:.*norm$",
    # ignore anything not in language_model
    "re:.*mm_projector.*",
    "re:.*vision.*",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=ignore,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        quant_config_key="text_config.quantization_config",
        ignore=ignore,
    ),
    max_workers=2,
    device="cuda:0",
)
