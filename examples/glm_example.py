# requires: einops, fla-core, tiktoken
from compressed_tensors.distributed import init_dist
from transformers import AutoModelForCausalLM, AutoProcessor, CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.utils import load_context
from llmcompressor.modeling.patch.kimi_k3_patch import patch_kimi_k3_ignore

# Small representative model with same MXFP4 quantization
MODEL_ID = "zai-org/GLM-5.3-Flash"

# Load model with the modified quantization config and disk offloading
init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={},
        offload_folder="/data/kylesayrs/hub/offload_folder",
    )
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

recipe = [
    REAPPruningModifier(sparsity=0.30, report_path="glm53_flash_report.pkl", prune=False),
    #QuantizationModifier(
    #    targets="re:.*block_sparse_moe.*",
    #    scheme="NVFP4",
    #    ignore=[
    #        "lm_head",
    #        r"re:.*block_sparse_moe\.gate",
    #        "re:.*vision_tower.*",
    #        "re:.*mlp_res_proj$",
    #        "re:.*routed_expert.*",
    #    ],
    #    weight_observer="nvfp4_expanded_mse",
    #),
]

oneshot(
    model=model,
    tokenizer=processor.tokenizer,
    dataset="perfectblend",
    splits=get_rank_partition("train", int(1024 * 2)),
    recipe=recipe,
    max_seq_length=2048,
    trust_remote_code_model=True,
    pipeline="sequential",
    batch_size=64,
    # The model is loaded pre-compressed (dequantize=False), so each subgraph must be
    # decompressed before calibration (otherwise the quantized forward hits a missing
    # `.weight`) and re-compressed afterwards to keep peak memory low.
    layerwise_decompression=False,
    layerwise_compression=False,
    shuffle_calibration_samples=False,
    # sequential_targets=["KimiMLAAttention", "KimiDeltaAttention", "ExpertMLPWithGate"],
    # sequential_targets_per_subgraph=300,
    propagate_error=False,
    sequential_targets_per_subgraph=3,
)

# SAVE_DIR = (
#     "/data/kylesayrs/hub/" + MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4-REAP10"
# )
# model.save_pretrained(SAVE_DIR)
# processor.save_pretrained(SAVE_DIR)
