from compressed_tensors.entrypoints.convert import FP8BlockDequantizer

from llmcompressor import model_free_ptq

MODEL_ID = "zai-org/GLM-5.3"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4"

# Keep these modules outside MXFP4 quantization.
ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*eh_proj$",
    "re:.*self_attn.indexer.weights_proj$",
    # wk IS fp8 in the source (dequantizer dequantizes it), but vLLM fuses
    # wk + weights_proj into a dense (quant_config=None) layer whose fp8-only
    # load path can't unpack mxfp4 weights; keep wk in bf16 instead
    "re:.*self_attn.indexer.wk$",
]

# Model-free PTQ processes checkpoint tensors directly, including MTP weights.
# This quantizes both the backbone and MTP to MXFP4, except the ignored modules.
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="MXFP4",
    ignore=ignore,
    converter=FP8BlockDequantizer.from_pretrained(MODEL_ID),
    device="cuda:0",
)
