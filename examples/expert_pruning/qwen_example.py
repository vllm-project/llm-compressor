# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import RazorPruningModifier
from llmcompressor.utils import load_context

# Select model and load it.
model_id = "Qwen/Qwen3.6-35B-A3B"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="disk", offload_folder="offload_folder"
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

recipe = RazorPruningModifier(sparsity=0.25, report_path="qwen3_report.json")

# Apply algorithms.
oneshot(
    model=model,
    dataset="razorcal",
    splits="train[:2048]",
    recipe=recipe,
    max_seq_length=2048,
    batch_size=32,
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-RAZOR25"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
