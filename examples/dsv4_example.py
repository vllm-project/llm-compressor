from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4PreTrainedModel,
)
from compressed_tensors.distributed import init_dist

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modeling.patch.deepseek_v4_patch import patch_dsv4_fp8_attention
from llmcompressor.utils import load_context

# Upstream BUG: norms should be loaded in float32, but usually aren't due to the base
# model having a quant_config which overrides this. Loading in float32 actually
# breaks the model definition (it expects bfloat16). Let's force load in bfloat16.
DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = set()
#qconfig = CompressedTensorsConfig(dequantize=False, use_optimized_inference=False)

# Select model and load it.
model_id = "/data/kylesayrs/hub/DeepSeek-V4-Flash-0731-ct"

init_dist()
# Keep the attention projections in FP8 through calibration: patch_dsv4_fp8_attention
# installs an FP8-aware attention forward so the compressed attention weights don't
# need to be decompressed (and don't hit the faulting fp8 Triton kernels).
with load_context(), patch_dsv4_fp8_attention():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=qconfig,
        device_map="auto_offload",
        max_memory={},
        offload_folder="/data/kylesayrs/hub/offload_folder",
    )

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure the quantization algorithm to run.
#   * quantize mlp/expert weights to NVFP4
#   * quantize attention projection weights to FP8_BLOCK
# model.model.layers.0.self_attn.q_a_proj
#
# wq_a  | q_a_proj
# wq_b  | q_b_proj
# wkv   | kv_proj
# wo_a  | o_a_proj
# wo_b  | o_b_proj

recipe = [
    REAPPruningModifier(sparsity=0.30, report_path="dsv4_flash.pkl", prune=False),
    GPTQModifier(
        config_groups={
            # "attention": QuantizationScheme(
            #     targets=[
            #         r"re:.*attn\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
            #         r"re:.*attn\.compressor\.indexer\.q_b_proj$",
            #     ],
            #     **FP8_BLOCK,
            # ),
            "experts": QuantizationScheme(
                targets=[
                    r"re:.*mlp\.experts.*",
                ],
                **NVFP4,
            ),
        },
        ignore=[],
    )
]

# Apply algorithms.
# due to the large size of DeepSeek-V4, we specify sequential targets such that
# only one block is loaded into GPU memory at a time
oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset="perfectblend",
    splits=get_rank_partition("train", int(1024 * 2)),
    recipe=recipe,
    max_seq_length=2048,
    pipeline="sequential",
    batch_size=64,
    # The model is loaded pre-compressed (dequantize=False), so each subgraph must be
    # decompressed before calibration (otherwise the quantized forward hits a missing
    # `.weight`) and re-compressed afterwards to keep peak memory low.
    layerwise_decompression=True,
    layerwise_compression=True,
    shuffle_calibration_samples=False,
    # sequential_targets=["KimiMLAAttention", "KimiDeltaAttention", "ExpertMLPWithGate"],
    # sequential_targets_per_subgraph=300,
    propagate_error=False,
    # sequential_targets_per_subgraph=3,
)

# Save to disk compressed.
SAVE_DIR = "/data/kylesayrs/hub/" + model_id.rstrip("/").split("/")[-1] + "-NVFP4-FP8-BLOCK"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
