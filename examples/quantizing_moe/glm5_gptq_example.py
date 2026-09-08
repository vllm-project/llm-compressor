# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import load_context

# Select model and load it.
# load_context() handles MoE linearization (converts fused 3D expert weights to
# per-expert 2D format) and CT offloading (converts accelerate hooks to
# compressed-tensors offload, compatible with calibration hooks).
# device_map="auto" places weights across available GPUs/CPU automatically.
model_id = "zai-org/GLM-5.2"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto_offload")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Layers 0-2 are dense; skip them and the output head.
ignore = [
    "re:model.layers.0.*",
    "re:model.layers.1.*",
    "re:model.layers.2.*",
    "lm_head",
    "re:.*mlp.gate$",
]

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)

# Two-target strategy:
#   - "GlmMoeDsaAttention": keeps GlmMoeDsaIndexer (data-dependent top-k control flow)
#     inside a leaf so the fx tracer never enters it.
#   - "ExpertMLP": each linearized expert is its own subgraph boundary so only one
#     expert's Hessian is resident in GPU memory at a time.
#
# sequential_targets_per_subgraph batches multiple ExpertMLP modules per subgraph,
# balancing memory usage against calibration runtime.
# Value = num_experts // 4 + buffer  (384 // 4 + 10 = 106 for GLM-5.2).
num_experts = getattr(model.config, "n_routed_experts", 384)
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    batch_size=4,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    sequential_targets=["GlmMoeDsaAttention", "ExpertMLP"],
    sequential_targets_per_subgraph=(num_experts // 4 + 10),
)

SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
