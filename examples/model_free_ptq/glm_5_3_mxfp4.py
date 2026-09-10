from compressed_tensors.entrypoints.convert import FP8BlockDequantizer

from llmcompressor import model_free_ptq


MODEL_ID = "zai-org/GLM-5.3"
SAVE_DIR = "/mnt/nvme_stripe/playground/dsikka/" + MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4"

ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*embed_tokens$",
    # not fp8-block quantized in the source checkpoint
    "re:.*eh_proj$",
    "re:.*self_attn.indexer.weights_proj$",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="MXFP4",
    ignore=ignore,
    converter=FP8BlockDequantizer(
        ignore=ignore,
    ),
    max_workers=2,
    device="cuda:0",
)