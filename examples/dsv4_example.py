from compressed_tensors.compressors import decompress_module
from compressed_tensors.distributed import init_dist
from compressed_tensors.quantization.quant_scheme import NVFP4, QuantizationScheme
from compressed_tensors.quantization.utils import is_module_quantized
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modeling.patch.deepseek_v4_patch import (
    FP8DeepseekV4Attention,
    install_dsv4_chat_template,
    patch_dsv4_config,
    patch_dsv4_fp8_attention,
)
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

# Select model and load it. This example starts from a compressed-tensors
# checkpoint: the attention projections and shared experts are FP8-BLOCK
# quantized and the routed experts are NVFP4 quantized.
MODEL_ID = "RedHatAI/DeepSeek-V4-Flash-0731-NVFP4"

# Load the model pre-compressed (dequantize=False):
#   * patch_dsv4_config rewrites the checkpoint's quantization targets (which use
#     pre-linearization weight names) to the module names of the loaded model
#   * patch_dsv4_fp8_attention installs an FP8-aware attention forward so the
#     compressed attention weights don't need to be decompressed (and don't hit
#     the fp8 Triton kernels, which fault on some accelerators)
init_dist()
qconfig = CompressedTensorsConfig(dequantize=False, use_optimized_inference=False)
with load_context(), patch_dsv4_config(), patch_dsv4_fp8_attention():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=qconfig,
        device_map="auto_offload",
        max_memory={},
        offload_folder="offload_folder",
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# DeepSeek-V4 has no Jinja chat template, so install one backed by the official
# checkpoint's `encode_messages` helper. This renders conversations in the format
# the model was trained on.
install_dsv4_chat_template(
    tokenizer, "deepseek-ai/DeepSeek-V4-Flash-0731", thinking_mode="chat"
)

# Decompress everything except the attention projections: the packed expert
# weights cannot run eagerly and are re-quantized during calibration, while the
# FP8 attention weights stay compressed and run through the FP8-aware attention
# forward installed by patch_dsv4_fp8_attention
attn_modules = {
    submodule
    for attn in model.modules()
    if isinstance(attn, FP8DeepseekV4Attention)
    for submodule in attn.modules()
}
for _, module in model.named_modules():
    if is_module_quantized(module) and module not in attn_modules:
        decompress_module(module)

# Configure the quantization algorithm to run.
#   * quantize mlp/expert weights to NVFP4
#   * the attention projections are already FP8-BLOCK compressed and are left as-is
#
# wq_a  | q_a_proj
# wq_b  | q_b_proj
# wkv   | kv_proj
# wo_a  | o_a_proj
# wo_b  | o_b_proj

recipe = GPTQModifier(
    config_groups={
        "experts": QuantizationScheme(
            targets=[
                r"re:.*mlp\.experts.*",
            ],
            **NVFP4,
        ),
    },
    ignore=[],
)

# Apply algorithms.
# due to the large size of DeepSeek-V4, we specify sequential targets such that
# only one block is loaded into GPU memory at a time
oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset="ultrachat_200k",
    splits=get_rank_partition("train_sft", 512),
    recipe=recipe,
    max_seq_length=2048,
    pipeline="sequential",
    batch_size=16,
    shuffle_calibration_samples=False,
    propagate_error=False,
)

# Save to disk compressed. The attention projections are still stored as FP8 with
# per-block scales, and the experts are re-packed as NVFP4.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-REQUANTIZED"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
