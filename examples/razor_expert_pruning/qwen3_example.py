# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import RazorPruningModifier

# Select model and load it.
model_id = "Qwen/Qwen3-30B-A3B-Thinking-2507"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prune the 25% least replaceable experts from every layer (128 -> 96
# experts), scoring each expert by the damage its deletion would do to the
# routed mixture output. Per-expert statistics are written to
# qwen3_report.json along the way, which can be visualized with
# tools/plot_reap_report.py or fed to ExpertPruner to prune any sparsity
# directly from the safetensors checkpoint (see
# examples/reap_expert_pruning/qwen3_prune_from_report.py).
recipe = RazorPruningModifier(sparsity=0.25, report_path="qwen3_report.json")

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:1024]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=1024,
    moe_calibrate_all_experts=False,  # only routed experts are evaluated
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-RAZOR-25"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
