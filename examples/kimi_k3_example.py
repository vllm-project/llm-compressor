# requires: einops, fla-core, tiktoken
from compressed_tensors.distributed import init_dist
from transformers import AutoConfig, AutoProcessor, CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.utils import load_context

# Small representative model with same MXFP4 quantization
MODEL_ID = "moonshotai/Kimi-K3"
# MODEL_ID = "inference-optimization/Kimi-K3-0.40B-MXFP4"

# Patch quantization config to
# 1. Fix an incomplete ignore list provided by the base checkpoint
# 2. Disable decompression (for later step)
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
qconfig = CompressedTensorsConfig(
    **config.quantization_config, dequantize=False, use_optimized_inference=False
)
qconfig.quantization_config.ignore += [
    "re:.*mlp_res_proj.*",
    "re:.*self_attention_res_proj.*",
    "re:.*routed_expert.*",
    "re:.*output_attn_res_proj.*",
]

# Load model with the modified quantization config and disk offloading
init_dist()
with load_context(KimiK3ForConditionalGeneration):
    model = KimiK3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=qconfig,
        device_map="auto_offload",
        trust_remote_code=True,
        max_memory={},
        offload_folder="/data/kylesayrs/hub/offload_folder",
    )
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

recipe = [
    REAPPruningModifier(sparsity=0.10),
    GPTQModifier(
        targets="re:.*block_sparse_moe.*",
        scheme="NVFP4",
        ignore=[
            "lm_head",
            r"re:.*block_sparse_moe\.gate",
            "re:.*vision_tower.*",
            "re:.*mlp_res_proj$",
            "re:.*routed_expert.*",
        ],
    ),
]

oneshot(
    model=model,
    tokenizer=processor.tokenizer,
    dataset="perfectblend",
    splits=get_rank_partition("train", 1024),
    recipe=recipe,
    max_seq_length=2048,
    trust_remote_code_model=True,
    pipeline="sequential",
    batch_size=2,
    layerwise_decompression=True,
    layerwise_compression=True,
)

SAVE_DIR = (
    "/data/kylesayrs/hub/" + MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-REAP10-GPTQ"
)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
