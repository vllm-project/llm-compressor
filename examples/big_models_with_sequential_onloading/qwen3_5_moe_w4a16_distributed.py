# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
#############################################################################
# Distributed W4A16 GPTQ quantization for Qwen3.5-122B-A10B (MoE, 8×GPU).
#
# Run with:
#   torchrun --nproc_per_node=8 qwen3_5_moe_w4a16_distributed.py
#
# Key differences from the dense-model example (llama3_8b_w8a8_distributed.py):
#   1. GPTQModifier (W4A16 weight-only) instead of QuantizationModifier
#   2. CPUCache instead of DistributedCPUCache for MoE expert tensors
#   3. actorder=None to avoid column-permutation races on IPC-shared pages
#
# Why CPUCache instead of DistributedCPUCache for MoE?
#   DistributedCPUCache.offload() creates one POSIX shared-memory file per
#   tensor and two collective ops per tensor (broadcast_object_list + barrier).
#   For 48 layers × 256 experts × 3 projections = 36,864 tensors this means
#   ~216 GB of /dev/shm usage and 73,728 blocking collectives during model load,
#   which is physically impractical. CPUCache stores each tensor in regular
#   pinned CPU memory with no shared-memory overhead.
#   broadcast_qparams_and_cleanup (PR #3066) handles the explicit writeback
#   needed to synchronise quantization parameters across independent CPUCaches.
#############################################################################

import torch
import torch.multiprocessing as torch_mp
from compressed_tensors.distributed import init_dist
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.cache.cpu import CPUCache
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

# file_system IPC: expert weight pages are shared across ranks via tmpfs
# mappings (zero-copy), avoiding descriptor-limit issues with fd-based shm.
torch_mp.set_sharing_strategy("file_system")

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"

init_dist()

# Override OffloadCache.cls_from_device to return CPUCache for CPU offload in
# distributed context. DistributedCPUCache is correct for dense models, but
# creates 36,864+ /dev/shm files (~216 GB) for this MoE model, exhausting
# the tmpfs before GPTQ calibration can begin.
_orig_cls_from_device = OffloadCache.cls_from_device.__func__


@classmethod
def _moe_cls_from_device(cls, device=None):
    cache_cls = _orig_cls_from_device(cls, device)
    # Downgrade DistributedCPUCache → CPUCache for MoE scale.
    # broadcast_qparams_and_cleanup writes quantization parameters back to
    # each rank's independent CPUCache after the distributed broadcast.
    from compressed_tensors.offload.cache.dist_cpu import DistributedCPUCache

    if cache_cls is DistributedCPUCache:
        return CPUCache
    return cache_cls


OffloadCache.cls_from_device = _moe_cls_from_device

with load_context(Qwen3_5MoeForConditionalGeneration):
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        torch_dtype="auto",
        trust_remote_code=True,
    )

# Restore original cls_from_device after model loading.
OffloadCache.cls_from_device = _orig_cls_from_device

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "re:.*lm_head$",
            "re:visual.*",
            "re:model.visual.*",
            "re:.*embed_tokens$",
            "re:.*mlp\\.gate$",
            "re:.*mlp\\.shared_expert_gate$",
            "re:.*linear_attn.*",
            "re:mtp.*",
        ],
        # actorder=None: GPTQ activation ordering reorders weight columns
        # in-place. Expert weights are IPC-shared physical pages; concurrent
        # column permutations from different ranks corrupt the tensor.
        actorder=None,
    )
]

oneshot(
    model=model,
    dataset="open_platypus",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

print("Saving...")
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-GPTQ-DDP"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
