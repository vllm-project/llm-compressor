# requires: einops, fla-core, tiktoken
from compressed_tensors.distributed import init_dist
from transformers import AutoConfig, AutoProcessor, CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "inference-optimization/Kimi-K3-0.40B-MXFP4"  # "moonshotai/Kimi-K3"

# Patch quantization config to
# 1. Fix an incomplete ignore list provided by the base checkpoint
# 2. Unfront dequantize modules for subsequent calibration/compression
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
qconfig = CompressedTensorsConfig(**config.quantization_config, dequantize=True)
qconfig.quantization_config.ignore += [
    "re:.*mlp_res_proj.*",
    "re:.*self_attention_res_proj.*",
    "re:.*routed_expert.*",
    "re:.*output_attn_res_proj.*",
]

# Load model with the modified quantization config
init_dist()
with load_context(KimiK3ForConditionalGeneration):
    model = KimiK3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=qconfig,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        r"re:.*block_sparse_moe\.gate",
        "re:.*vision_tower.*",
    ],
)

oneshot(
    model=model,
    tokenizer=processor.tokenizer,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    trust_remote_code_model=True,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
