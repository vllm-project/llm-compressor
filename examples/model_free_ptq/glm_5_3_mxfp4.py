from compressed_tensors.entrypoints.convert import FP8BlockDequantizer

from llmcompressor import model_free_ptq

MODEL_ID = "zai-org/GLM-5.3"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4"

# modules that were not fp8-block quantized in the source checkpoint
ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*eh_proj$",
    "re:.*self_attn.indexer.weights_proj$",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="MXFP4",
    # wk IS fp8 in the source (dequantizer dequantizes it), but vLLM fuses
    # wk + weights_proj into a dense (quant_config=None) layer whose fp8-only
    # load path can't unpack mxfp4 weights — keep wk in bf16 instead
    ignore=ignore + ["re:.*self_attn.indexer.wk$"],
    converter=FP8BlockDequantizer(ignore=ignore),
    max_workers=2,
    device="cuda:0",
)
