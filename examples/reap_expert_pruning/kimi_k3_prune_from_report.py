# Prune experts from the RedHatAI Kimi-K3 NVFP4 checkpoint using a REAP
# saliency report.
#
# The report is produced ahead of time by calibrating the model with
# `REAPPruningModifier(prune=False, report_path="kimi_swe_report.json")`
# (see examples/kimi_k3_example.py). Pruning then runs directly on the
# safetensors checkpoint, shard by shard, without loading the model into
# memory. This makes it cheap to try several sparsities / metrics from a
# single calibration run, and it preserves the NVFP4 format: each expert's
# packed weights and qparams (weight_packed / weight_scale /
# weight_global_scale / input_global_scale) are kept or dropped together.
from compressed_tensors.entrypoints.convert import convert_checkpoint

from llmcompressor.entrypoints.converter import ExpertPruner

MODEL_ID = "RedHatAI/Kimi-K3-NVFP4"
REPORT_PATH = "kimi_swe_report.json"
SPARSITY = 0.10

# Kimi-K3 (DeepSeek-V3-style) MoE tensors live under
# `language_model.model.layers.<i>.block_sparse_moe` for layers 1..92 (layer 0
# is dense, `first_k_dense_replace=1`), with 896 experts per layer routed
# top-16:
#   router weight:  ...block_sparse_moe.gate.weight
#   routing bias:   ...block_sparse_moe.gate.e_score_correction_bias
#                    (noaux_tc bias, one entry per expert, must be sliced
#                    alongside the router rows)
#   expert weights: ...block_sparse_moe.experts.<e>.w1.weight_packed  (gate)
#                    ...block_sparse_moe.experts.<e>.w2.weight_packed  (down)
#                    ...block_sparse_moe.experts.<e>.w3.weight_packed  (up)
#                    plus per-tensor weight_scale / weight_global_scale /
#                    input_global_scale qparams, all covered by the same
#                    expert_pattern
# The shared experts (`block_sparse_moe.shared_experts.*`) and the latent-MoE
# projections (`block_sparse_moe.routed_expert_*`) are not routed experts and
# are left untouched, as is the vision tower. Kimi-K3 has no static
# hash-routing tables, so index_pattern=None.
#
# With uniform=True, round(SPARSITY * 896) = 90 experts are pruned from every
# MoE layer (896 -> 806) and the new count is written to
# text_config.num_experts. Note that non-uniform pruning is not recommended
# here: the config stores the per-token routing count as
# `num_experts_per_token`, which the pruner's floor detection does not read,
# so the per-layer expert count could drop below the top-16 routing floor.
pruner = ExpertPruner.from_pretrained(
    MODEL_ID,
    sparsity=SPARSITY,
    metric="saliency",
    uniform=True,
    saliency_report_path=REPORT_PATH,
    router_pattern=r"block_sparse_moe\.gate\.weight$",
    expert_pattern=r"block_sparse_moe\.experts\.\d+\.(w1|w2|w3)",
    aux_pattern=r"block_sparse_moe\.gate\.e_score_correction_bias$",
    index_pattern=None,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + f"-REAP-{int(SPARSITY * 100)}"
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=pruner,
    max_workers=8,
)
