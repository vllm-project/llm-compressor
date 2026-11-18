# Prune experts from the
# inference-optimization/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-compressed-tensors
# checkpoint using a RAZOR saliency report.
#
# The report is produced ahead of time by calibrating with
# `RazorPruningModifier(sparsity=0.0, report_path="nemotron35_super_vl_report.json")`
# (see nemotron35_super_vl_example.py). Pruning then runs directly on the
# safetensors checkpoint, shard by shard, without loading the model into
# memory. This makes it cheap to try several sparsities / metrics from a single
# calibration run, and it preserves the checkpoint's NVFP4 weights exactly:
# only expert tensors are dropped/renumbered and router rows sliced.
import os

from compressed_tensors.entrypoints.convert import convert_checkpoint
from huggingface_hub.constants import HF_HUB_CACHE

from llmcompressor.entrypoints.converter import ExpertPruner

MODEL_ID = (
    "inference-optimization/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-"
    "NVFP4-compressed-tensors"
)
REPORT_PATH = "nemotron35_super_vl_report.json"
SPARSITY = 0.20

# The checkpoint stores its language model under `language_model.backbone.*`
# (Megatron naming): 88 backbone layers, of which the `moe` ones carry a
# `mixer` with 512 routed experts behind a `mixer.gate` router. The MTP layers
# live under a separate `language_model.mtp.*` prefix, which these patterns do
# not match: MTP calibration is not supported, so those layers were not in the
# report and are left untouched. The router's per-expert
# `e_score_correction_bias` is sliced along with the router weight.
BACKBONE = r"language_model\.backbone\.layers\.\d+\."

# Prune the 20% least replaceable experts across all backbone MoE layers (512
# -> ~410 experts per layer).
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
    router_pattern=BACKBONE + r"mixer\.gate\.weight$",
    expert_pattern=BACKBONE + r"mixer\.experts\.\d+\.(up|down)_proj",
    # Nemotron-3.5 routes with a per-expert `e_score_correction_bias`, which
    # must be sliced along with the router weight
    aux_pattern=BACKBONE + r"mixer\.gate\.e_score_correction_bias$",
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
