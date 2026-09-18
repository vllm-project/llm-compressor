# requires: einops, fla-core, tiktoken
import torch.distributed as dist
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.distributed import init_dist
from transformers import AutoConfig, AutoProcessor, CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.modeling.patch.kimi_k3_patch import patch_kimi_k3_ignore
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Small representative model with same MXFP4 quantization
MODEL_ID = "inference-optimization/Kimi-K3-0.40B-MXFP4"  # "moonshotai/Kimi-K3"

# Patch quantization config to disable quantization (for a later step)
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
qconfig = CompressedTensorsConfig(dequantize=False, use_optimized_inference=False)

# Load model with the modified quantization config and disk offloading
# Patch an incomplete ignore list provided by the base checkpoint
init_dist()
with load_context(KimiK3ForConditionalGeneration), patch_kimi_k3_ignore():
    model = KimiK3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=qconfig,
        device_map="auto_offload",
        trust_remote_code=True,
        max_memory={},
        offload_folder="offload_folder",
    )
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Decompress model upfront before calibrating
ModelCompressor.from_pretrained_model(model).decompress_model(model)

recipe = QuantizationModifier(
    targets="re:.*block_sparse_moe.*",
    scheme="NVFP4",
    ignore=[
        r"re:.*block_sparse_moe\.gate",
        "re:.*routed_expert.*",
        "re:.*mlp_res_proj$",
        "re:.*vision_tower.*",
        "lm_head",
    ],
    weight_observer="nvfp4_expanded_mse",
)

oneshot(
    model=model,
    tokenizer=processor.tokenizer,
    dataset="perfectblend",
    splits=get_rank_partition("train", 512),
    recipe=recipe,
    max_seq_length=2048,
    trust_remote_code_model=True,
    batch_size=16,
    shuffle_calibration_samples=False,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

dist.destroy_process_group()
