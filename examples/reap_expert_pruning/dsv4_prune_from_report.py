# Prune experts from a Qwen3 MoE checkpoint using a REAP saliency report.
#
# The report is produced ahead of time by calibrating with
# `REAPPruningModifier(prune=False, report_path="qwen3_report.json")`
# (see qwen3_example.py). Pruning then runs directly on the safetensors
# checkpoint, shard by shard, without loading the model into memory. This makes
# it cheap to try several sparsities / metrics from a single calibration run.
from compressed_tensors.entrypoints.convert import convert_checkpoint

from llmcompressor.entrypoints.converter import ExpertPruner

MODEL_ID = "RedHatAI/DeepSeek-V4-Flash-0731-NVFP4"
REPORT_PATH = "dsv4_swe_report.json"
SPARSITY = 0.25

# Prune the 25% least salient experts from every layer (128 -> 96 experts).
#   metric="saliency":           rank experts by raw REAP saliency
#   metric="layerwise_saliency": saliency normalized by each layer's maximum,
#                                useful for comparing layers when uniform=False
#   metric="count":              rank experts by how many tokens they received
#   metric="magnitude":          rank experts by router weight magnitude (no
#                                report required)
# With uniform=False, the 25% least salient experts across all layers are pruned
# instead, so layers may retain different numbers of experts. These per-layer
# counts are recorded in `quantization_config.layer_overrides`.
#
# DeepSeek-V4 carries two auxiliary router tensors per layer that must be pruned
# alongside the router weight (handled by the default aux_pattern / index_pattern):
#   ffn.gate.bias     [num_experts]            routing bias, sliced by retained experts
#   ffn.gate.tid2eid  [vocab, num_experts_per_tok]
#                     hash-routing table whose VALUES are expert indices, remapped
#                     to the renumbered experts. The first `num_hash_layers` layers
#                     route statically via this table; uniform=False protects those
#                     layers from pruning (uniform=True would raise on them).
pruner = ExpertPruner.from_pretrained(
    MODEL_ID,
    sparsity=SPARSITY,
    metric="layerwise_saliency",
    uniform=False,
    saliency_report_path=REPORT_PATH,
    router_pattern=r"ffn\.gate\.weight",
    expert_pattern=r"ffn\.experts\.\d+\.(w1|w2|w3)",
)

SAVE_DIR = (
    "/data/kylesayrs/"
    + MODEL_ID.rstrip("/").split("/")[-1]
    + f"-REAP-{int(SPARSITY * 100)}-swe-nonuniform"
)
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=pruner,
    max_workers=8,
)
