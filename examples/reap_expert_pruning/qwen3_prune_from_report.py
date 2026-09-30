# Prune experts from a Qwen3 MoE checkpoint using a REAP saliency report.
#
# The report is produced ahead of time by calibrating with
# `REAPPruningModifier(prune=False, report_path="qwen3_report.json")`
# (see qwen3_example.py). Pruning then runs directly on the safetensors
# checkpoint, shard by shard, without loading the model into memory. This makes
# it cheap to try several sparsities / metrics from a single calibration run.
from compressed_tensors.entrypoints.convert import convert_checkpoint

from llmcompressor.entrypoints.converter import ExpertPruner

MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
REPORT_PATH = "qwen3_report.json"
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
pruner = ExpertPruner.from_pretrained(
    MODEL_ID,
    sparsity=SPARSITY,
    metric="layerwise_saliency",
    uniform=False,
    saliency_report_path=REPORT_PATH,
    # Qwen3 MoE stores one tensor per expert, e.g.
    # `model.layers.0.mlp.experts.3.gate_proj.weight`
    expert_pattern=r"mlp\.experts\.\d+\.(gate|up|down)_proj",
)

SAVE_DIR = (
    "/root/.cache/huggingface/"
    + MODEL_ID.rstrip("/").split("/")[-1]
    + f"-REAP-{int(SPARSITY * 100)}-nonuniform"
)
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=pruner,
    max_workers=8,
)
