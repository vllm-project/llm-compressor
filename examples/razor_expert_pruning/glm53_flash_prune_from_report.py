# Prune experts from the RedHatAI/GLM-5.3-Flash-NVFP4 checkpoint using a RAZOR
# saliency report.
#
# The report is produced ahead of time by calibrating with
# `RazorPruningModifier(sparsity=0.0, report_path="glm53_flash_report.json")`
# (see glm53_flash_example.py). Pruning then runs directly on the safetensors
# checkpoint, shard by shard, without loading the model into memory. This makes
# it cheap to try several sparsities / metrics from a single calibration run.
import os

from compressed_tensors.entrypoints.convert import convert_checkpoint
from huggingface_hub.constants import HF_HUB_CACHE

from llmcompressor.entrypoints.converter import ExpertPruner

MODEL_ID = "RedHatAI/GLM-5.3-Flash-NVFP4"
REPORT_PATH = "glm53_flash_report.json"
SPARSITY = 0.20

# The checkpoint carries one MTP layer at `layers.45`, one past the 45 backbone
# decoder layers (of which layers 3-44 are MoE with 288 experts each). It was
# not calibrated (glm53_flash_example.py loads the model with
# `load_mtp=False`), so the patterns below exclude it and it is left untouched.
# `uniform=False` matches that: the retained expert counts are recorded per
# layer in `quantization_config.moe_sparsity_config` while `n_routed_experts`
# keeps its original value of 288, which still matches the untouched MTP layer
# (with `uniform=True` the config would instead be rewritten to the pruned
# count of every backbone layer, contradicting the MTP layer's weights).
BACKBONE = r"language_model\.layers\.(?!45\.)\d+\."

# Prune the 20% least replaceable experts across all backbone layers (288 ->
# ~230 experts per layer).
#   metric="layerwise_saliency": saliency normalized by each layer's maximum,
#                                which makes scores comparable across layers
#                                when pruning non-uniformly
#   metric="saliency":           rank experts by raw RAZOR saliency
#   metric="count":              rank experts by how many tokens they received
#   metric="magnitude":          rank experts by router weight magnitude (no
#                                report required)
pruner = ExpertPruner.from_pretrained(
    MODEL_ID,
    sparsity=SPARSITY,
    metric="layerwise_saliency",
    uniform=False,
    saliency_report_path=REPORT_PATH,
    router_pattern=BACKBONE + r"mlp\.gate\.weight$",
    expert_pattern=BACKBONE + r"mlp\.experts\.\d+\.(gate|up|down)_proj",
    # GLM-5.3 routes with a per-expert `e_score_correction_bias`, which must be
    # sliced along with the router weight (the default `aux_pattern` instead
    # matches DeepSeek's `ffn.gate.bias`)
    aux_pattern=BACKBONE + r"mlp\.gate\.e_score_correction_bias$",
)

SAVE_DIR = os.path.join(
    HF_HUB_CACHE,
    MODEL_ID.rstrip("/").split("/")[-1] + f"-Sparse{int(SPARSITY * 100)}",
)
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=pruner,
    max_workers=8,
)
