# Calibrate NVIDIA's Nemotron-3.5-Super-VL-120B-A12B (BF16 release) with RAZOR
# expert pruning in report-only mode. Run with plain python (a single process):
# the model is ~240 GB in bf16, so it is disk-offloaded and calibrated layer by
# layer.
#
# Nemotron-3.5's MoE layers are *latent* MoEs: each block projects its input
# down to moe_latent_size (1024) for the experts while the router keeps scoring
# the full hidden size (4096). RazorPruningModifier resolves the routing
# context from the activations the router itself consumed (captured by a
# pre-hook on the router), evaluates the experts in their latent space, and
# reads the router's sigmoid score function, so these layers are scored like
# any other MoE.
from transformers import AutoModelForImageTextToText, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import RazorPruningModifier
from llmcompressor.utils import load_context

# The saliency report is calibrated on the BF16 release of the model. The
# NVFP4-compressed-tensors release (see
# nemotron35_super_vl_prune_from_report.py for its id) cannot be loaded for
# calibration yet: it is a remote-code vision-language composition for which no
# direct 2D expert load mapping is registered, and its quantization config
# lists exact module paths in checkpoint naming
# (language_model.backbone.*), which do not match the module tree transformers
# builds (language_model.model.*). The routing statistics are shared between
# the two releases -- the router, the latent projections, and the shared
# experts are unquantized in the NVFP4 checkpoint -- so the report transfers;
# only the expert outputs carry NVFP4 quantization noise. Pruning then runs
# directly on the NVFP4 checkpoint's safetensors (see
# nemotron35_super_vl_prune_from_report.py), whose quantized weights are left
# untouched.
MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
REPORT_PATH = "nemotron35_super_vl_report.json"

with load_context(AutoModelForImageTextToText, load_mtp=False):
    # Nemotron-3.5-Super-VL is a custom-code architecture (nemotron_h_omni)
    # built around a NemotronH language model, so it must be loaded with its
    # multimodal class (plus trust_remote_code) to keep the vision tower.
    # `load_mtp=False` leaves the MTP layers (language_model.mtp.*) unloaded:
    # MTP calibration is not supported, so they are excluded from the report
    # and stay untouched by any pruning done from that report.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto_offload",
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Score each expert by the damage its deletion would do to the routed mixture
# output. With sparsity=0.0 the model is left untouched and only per-expert
# statistics (routed-token count and saliency, one inner list per MoE layer,
# ordered by layer index) are written to nemotron35_super_vl_report.json, which
# can be visualized with tools/plot_reap_report.py or fed to ExpertPruner to
# prune any sparsity directly from the NVFP4 safetensors checkpoint (see
# nemotron35_super_vl_prune_from_report.py).
recipe = RazorPruningModifier(sparsity=0.0, report_path=REPORT_PATH)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    dataset="razorcal",
    splits="train[:4096]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=4096,
    moe_calibrate_all_experts=False,  # only routed experts are evaluated
    # the report-only recipe never modifies the model, so a second
    # (hook-free) propagation pass over the calibration data would recompute
    # identical outputs; skip it
    propagate_error=False,
)

# The model is not saved: it was left untouched by the report-only recipe, and
# the report is all that nemotron35_super_vl_prune_from_report.py needs to
# prune the NVFP4-compressed-tensors checkpoint.
