# Prune experts from an OLMoE checkpoint with AIMER.
#
# AIMER scores every expert from its weights alone, so no calibration data is
# needed and the model is never loaded: pruning runs directly on the safetensors
# checkpoint, shard by shard.
from compressed_tensors.entrypoints.convert import convert_checkpoint

from llmcompressor.entrypoints.converter import AIMERExpertPruner

MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"
SPARSITY = 0.25

# Prune the 25% of experts with the highest AIMER score from every layer
# (64 -> 48 experts). With uniform=False, the highest-scoring experts across all
# layers are pruned instead, so layers may retain different numbers of experts.
pruner = AIMERExpertPruner.from_pretrained(
    MODEL_ID,
    sparsity=SPARSITY,
    # OLMoE stores one tensor per expert, e.g.
    # `model.layers.0.mlp.experts.3.gate_proj.weight`
    expert_pattern=r"mlp\.experts\.\d+\.(gate|up|down)_proj",
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + f"-AIMER-{int(SPARSITY * 100)}"
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=pruner,
    max_workers=8,
)
