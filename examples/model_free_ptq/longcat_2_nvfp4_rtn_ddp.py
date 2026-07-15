"""
LongCat-2.0 NVFP4 (W4A4) Quantization - DDP Version

Quantizes LongCat-2.0 to NVFP4 (4-bit weights, 4-bit activations) via oneshot with DDP.
Uses RTN for weights and calibration data for activations.

This is more aggressive than NVFP4A16:
- NVFP4A16: 4-bit weights, 16-bit activations (~800GB-1TB)
- NVFP4: 4-bit weights, 4-bit activations (~400-600GB)

Trade-offs:
- Pros: ~2x smaller memory footprint, can run on fewer GPUs
- Cons: More accuracy degradation due to quantized activations

Model will fit on:
- 4-5 B200s (144GB) or B300s (192GB)
- 6-8 H100s/H200s (80-141GB)

Usage:
    torchrun --nproc_per_node=8 longcat_2_nvfp4_rtn_ddp.py
"""

import os
import torch
from compressed_tensors.offload import dispatch_model, init_dist
from datasets import load_dataset
from transformers import AutoTokenizer, LongcatFlashForCausalLM, AutoConfig
from transformers.configuration_utils import PretrainedConfig
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.utils import load_context
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# LongcatFlashConfig with oe_* -> ngram_*/emb_* aliasing support
# (vendored from SGLang to avoid heavy dependencies)
class LongcatFlashConfig(PretrainedConfig):
    model_type = "longcat_flash"

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=6144,
        ffn_hidden_size=12288,
        expert_ffn_hidden_size=2048,
        num_layers=28,
        num_hidden_layers=None,
        num_attention_heads=64,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=128,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_routed_experts=512,
        moe_topk=12,
        max_position_embeddings=131072,
        rms_norm_eps=1e-05,
        rope_theta=10000000.0,
        rope_scaling=None,
        routed_scaling_factor=6.0,
        zero_expert_num=256,
        ngram_vocab_size_ratio=None,
        emb_neighbor_num=None,
        emb_split_num=None,
        oe_vocab_size_ratio=None,
        oe_neighbor_num=None,
        oe_split_num=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else num_layers
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_intermediate_size = expert_ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_routed_experts = n_routed_experts
        self.moe_topk = moe_topk
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.routed_scaling_factor = float(routed_scaling_factor) if routed_scaling_factor is not None else 6.0
        self.zero_expert_num = zero_expert_num

        # Handle oe_* -> ngram_*/emb_* aliasing for LongCat-2.0 compatibility
        if ngram_vocab_size_ratio is None:
            ngram_vocab_size_ratio = oe_vocab_size_ratio
        if emb_neighbor_num is None:
            emb_neighbor_num = oe_neighbor_num
        if emb_split_num is None:
            emb_split_num = oe_split_num

        self.oe_vocab_size_ratio = oe_vocab_size_ratio
        self.oe_neighbor_num = oe_neighbor_num
        self.oe_split_num = oe_split_num
        self.use_ngram_embedding = ngram_vocab_size_ratio is not None
        if self.use_ngram_embedding:
            self.ngram_embedding_m = int(ngram_vocab_size_ratio * vocab_size)
            self.ngram_embedding_n = emb_neighbor_num
            self.ngram_embedding_k = emb_split_num

# Register the config
AutoConfig.register("longcat_flash", LongcatFlashConfig, exist_ok=True)

# Model source - local path with config fixes for LongcatFlash compatibility
MODEL_ID = os.environ.get("MODEL_ID", "/home/HDCharles/hf_hub/models--meituan-longcat--LongCat-2.0")

# Save directory
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~"))
HF_HUB_DIR = os.environ.get("HF_HUB_DIR", HF_HOME)
SAVE_DIR = os.path.join(HF_HUB_DIR, "LongCat-2.0-NVFP4-RTN-DDP")

print(f"Loading model from: {MODEL_ID}")
print(f"Saving quantized model to: {SAVE_DIR}")
print()

# Initialize distributed training
init_dist()

# Load model using LongcatFlashForCausalLM
print("Loading model (this may take a while for large MoE models)...")
offload_folder = os.path.join(os.path.dirname(SAVE_DIR), "offload_tmp")
os.makedirs(offload_folder, exist_ok=True)
with load_context(LongcatFlashForCausalLM):
    model = LongcatFlashForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",  # Fit as much as possible on GPU, rest on CPU/disk
        offload_folder=offload_folder,  # Required for large MoE models
        max_memory={"cpu": 500e9},  # Don't exceed 500GB RAM
        ignore_mismatched_sizes=True,  # LongCat-2.0 has architecture differences from base LongcatFlash
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load calibration dataset for activation quantization
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 2048

print(f"Loading calibration dataset: {DATASET_ID}")
ds = load_dataset(DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES))
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure NVFP4 quantization
# NVFP4 quantizes both weights (RTN) and activations (requires calibration)
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        # Embedding and output layers
        "model.embed_tokens",
        "lm_head",

        # Normalization layers (1D tensors)
        "re:.*norm.*",

        # MoE routing gates (critical for expert selection)
        "re:.*gate$",
        "re:.*router.*",

        # MLA-specific layers that may be incompatible
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",

        # Multi-token prediction layers
        "re:.*mtp.*",
    ],
)

print("Applying NVFP4 quantization...")
print("- Weights: RTN (Round-To-Nearest)")
print("- Activations: calibrated with dataset")
print()

# Track performance
torch.accelerator.reset_peak_memory_stats()
import time
start_time = time.time()

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

elapsed_time = time.time() - start_time
peak_memory_gb = torch.accelerator.max_memory_allocated() / (1024**3)
print("Quantization Complete")
print(f"Time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello, I am LongCat-2.0", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save quantized model
print(f"Saving quantized model to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n✓ Quantization complete!")
print(f"\nModel saved to: {SAVE_DIR}")
print("\nExpected memory requirements for inference:")
print("- 4-5 B200s (144GB) or B300s (192GB)")
print("- 6-8 H100s/H200s (80-141GB)")
print("\nNote: NVFP4 (W4A4) is more aggressive than NVFP4A16 (W4A16)")
print("      Expect some accuracy degradation, but 2x memory reduction.")

# Clean up distributed process group
torch.distributed.destroy_process_group()
