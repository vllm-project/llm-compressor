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
#   73,728 blocking collectives during model load alone. /dev/shm is tmpfs backed
#   by physical RAM, so each shm file consumes physical RAM. On Qwen3.5-122B-A10B
#   (L20×8): /dev/shm grows from 231 GB (non-expert modules) to 351 GB (+120 GB)
#   before physical RAM is exhausted — the remaining expert tensors can never be
#   loaded. CPUCache stores each tensor in regular pinned CPU memory (no /dev/shm,
#   no per-tensor collectives). On Qwen3.5-35B-A3B (expert shm peaks at ~67 GB)
#   either mode works; CPUCache is only needed when expert shm would exceed
#   available physical RAM.
#   broadcast_qparams_and_cleanup (PR #3066) handles the explicit writeback
#   needed to synchronise quantization parameters across independent CPUCaches.
#############################################################################

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from compressed_tensors.distributed import init_dist
from compressed_tensors.offload import get_execution_device
from compressed_tensors.offload.cache.cpu import CPUCache
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils.dev import load_context, skip_weights_initialize

# file_system IPC: expert weight pages are shared across ranks via tmpfs
# mappings (zero-copy), avoiding descriptor-limit issues with fd-based shm.
torch_mp.set_sharing_strategy("file_system")
os.environ.setdefault("LINEARIZE_NO_COPY", "1")

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"

init_dist()
local_rank = int(os.environ["LOCAL_RANK"])


def _cpu_offload_module(module: torch.nn.Module, execution_device: str) -> None:
    """Wrap module parameters/buffers in CPUCache (pinned CPU, no /dev/shm)."""
    module._parameters = CPUCache.from_mapping(
        module._parameters,
        onload_device=execution_device,
        offload_device="cpu",
    )
    module._buffers = CPUCache.from_mapping(
        module._buffers,
        onload_device=execution_device,
        offload_device="cpu",
    )


@classmethod
@torch.no_grad()
def _ddp_from_experts_module(cls, experts, config):
    """Build LinearExperts2D with CPUCache from the already-broadcast 3D tensors.

    Called after the fused 3D expert tensors (gate_up_proj / down_proj) are
    already present on all ranks (broadcast happened at the 3D level), so
    CPUCache.from_mapping receives real tensors on every rank — not meta tensors.
    This is why we patch from_experts_module rather than cls_from_device: the
    cls_from_device path runs before tensors are broadcast and would fail on
    non-rank-0 ranks with 'Cannot copy out of meta tensor'.
    """
    with skip_weights_initialize():
        self = cls(config)

    intermediate_size = self.intermediate_size
    for index in range(self.num_experts):
        expert = self[index]
        if hasattr(expert, "gate_proj"):
            expert.gate_proj._parameters["weight"] = experts.gate_up_proj[
                index, :intermediate_size
            ]
            expert.up_proj._parameters["weight"] = experts.gate_up_proj[
                index, intermediate_size:
            ]
            expert.down_proj._parameters["weight"] = experts.down_proj[index]
        else:
            expert.up_proj._parameters["weight"] = experts.up_proj[index]
            expert.down_proj._parameters["weight"] = experts.down_proj[index]

    if dist.is_initialized():
        dist.barrier()

    execution_device = get_execution_device(experts, default=f"cuda:{local_rank}")
    for module in self.modules():
        if module._parameters:
            _cpu_offload_module(module, execution_device)
    return self


# Patch from_experts_module so expert Linear layers use CPUCache instead of
# DistributedCPUCache. This must be set before from_pretrained is called.
LinearExperts2D.from_experts_module = _ddp_from_experts_module

with load_context(Qwen3_5MoeForConditionalGeneration):
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        torch_dtype="auto",
        trust_remote_code=True,
    )

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
