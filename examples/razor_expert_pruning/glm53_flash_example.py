# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
#
# Calibrate RedHatAI/GLM-5.3-Flash-NVFP4 with RAZOR expert pruning in
# report-only mode. Run with plain python (a single process): the decompressed
# expert weights which accumulate over the run (~600 GB for this model) live in
# host RAM, which distributed runs would instead place in shared memory
# (/dev/shm), a much scarcer resource.
from transformers import AutoTokenizer, Glm5NextForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import RazorPruningModifier
from llmcompressor.utils import load_context

# Select model and load it.
MODEL_ID = "RedHatAI/GLM-5.3-Flash-NVFP4"
REPORT_PATH = "glm53_flash_report.json"

with load_context(Glm5NextForConditionalGeneration, load_mtp=False):
    # GLM-5.3-Flash is a vision-language MoE model, so it must be loaded with
    # its `Glm5NextForConditionalGeneration` class (not `AutoModelForCausalLM`)
    # in order to keep the vision tower. `load_mtp=False` leaves the MTP layer
    # (checkpoint `layers.45`, one past the 45 backbone decoder layers)
    # unloaded: MTP calibration is not supported, so it is excluded from the
    # report and stays untouched by any pruning done from that report.
    model = Glm5NextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Score each expert by the damage its deletion would do to the routed mixture
# output. With sparsity=0.0 the model is left untouched and only per-expert
# statistics (routed-token count and saliency, one inner list per MoE layer,
# ordered by layer index) are written to glm53_flash_report.json, which can be
# visualized with tools/plot_reap_report.py or fed to ExpertPruner to prune any
# sparsity directly from the safetensors checkpoint (see
# glm53_flash_prune_from_report.py).
recipe = RazorPruningModifier(sparsity=0.0, report_path=REPORT_PATH)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    dataset="perfectblend",
    splits="train[:4096]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=4096,
    moe_calibrate_all_experts=False,  # only routed experts are evaluated
    # the experts are already NVFP4-quantized in this checkpoint, so each layer
    # is decompressed (to dequantized weights) just before it is calibrated
    layerwise_decompression=True,
    # the report-only recipe never modifies the model, so a second
    # (hook-free) propagation pass over the calibration data would recompute
    # identical outputs; skip it
    propagate_error=False,
)

# The model is not saved: it was left untouched by the report-only recipe, and
# the report is all that glm53_flash_prune_from_report.py needs to prune the
# original checkpoint.
