from compressed_tensors.offload import init_dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.utils import load_context

# Select model and load it.
init_dist()
model_id = "Qwen/Qwen3-30B-A3B-Thinking-2507"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="disk", offload_folder="offload_folder"
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prune the 25% least salient experts from every layer (128 -> 96 experts).
# Per-expert statistics are written to qwen3_report.json along the way, which
# can be visualized with tools/plot_reap_report.py.
#
# To defer pruning instead, set sparsity=0.0: the model is left untouched and
# only the report is written, and qwen3_prune_from_report.py can then prune any
# sparsity directly from the safetensors checkpoint using that report.
recipe = REAPPruningModifier(sparsity=0.25, report_path="qwen3_report.json")

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:1024]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=1024,
    sequential_targets_per_subgraph=4,
    batch_size=16,
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-REAP-25"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
