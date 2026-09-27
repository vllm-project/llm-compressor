from compressed_tensors.offload import init_dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.utils import load_context

# Select model and load it.
init_dist()
model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="disk", offload_folder="offload_folder"
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prune 25% of the experts in each MoE layer, based on saliency.
# You can adjust this value to prune more or less aggressively.
recipe = REAPPruningModifier(prune=False, report_path="qwen3_report.json")

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    sequential_targets_per_subgraph=4,
    batch_size=16,
)
