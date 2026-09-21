#!/usr/bin/env python3
"""
Verification script: broadcast_qparams_and_cleanup writeback (PR #3066).

Demonstrates that CPUCache requires an explicit update_offload_parameter
writeback after dist.broadcast, and that the LinearExperts2D.from_experts_module
patch (PR #3202) alone does not prevent weight_scale corruption.

Three modes via environment variables:

  USE_ORIG_BROADCAST=0 (default)
      PR #3066 fix active. Expects SCALE_CHECK CLEAN on all ranks.

  USE_ORIG_BROADCAST=1
      Buggy broadcast: no CPUCache writeback. Expects SCALE_CHECK CORRUPTED.
      Simulates the state before PR #3066, with PR #3202 fix still applied.

  USE_DIST_CACHE=1
      Skip CPUCache patch; use DistributedCPUCache (maintainer-preferred for
      smaller models). WARNING: exhausts /dev/shm for 122B-scale MoE models.

Usage:
  # Confirm fix is working:
  torchrun --nproc_per_node=8 verify_broadcast_writeback.py

  # Reproduce the bug (no writeback, with PR #3202 model-load fix):
  USE_ORIG_BROADCAST=1 torchrun --nproc_per_node=8 verify_broadcast_writeback.py

  # Override model and calibration size:
  MODEL_PATH=Qwen/Qwen3.5-35B-A3B NUM_CALIBRATION_SAMPLES=16 \\
      torchrun --nproc_per_node=4 verify_broadcast_writeback.py
"""

import os
from functools import wraps  # noqa: F401

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from compressed_tensors.distributed import init_dist, wait_for_comms
from compressed_tensors.offload import get_execution_device
from compressed_tensors.offload.cache.cpu import CPUCache
from compressed_tensors.offload.dist_utils import as_broadcastable
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)

import llmcompressor.modifiers.gptq.base as _gptq_mod
import llmcompressor.utils.dist as _dist_mod
from llmcompressor import oneshot
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils.dev import load_context, skip_weights_initialize

torch_mp.set_sharing_strategy("file_system")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("LINEARIZE_NO_COPY", "1")

MODEL_ID = os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-122B-A10B")
NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_CALIBRATION_SAMPLES", "32"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", "1024"))
USE_ORIG_BROADCAST = os.environ.get("USE_ORIG_BROADCAST", "0") == "1"
USE_DIST_CACHE = os.environ.get("USE_DIST_CACHE", "0") == "1"

init_dist()
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])


def _log(msg):
    print(f"[verify:r{rank}] {msg}", flush=True)


def _log0(msg):
    if rank == 0:
        print(f"[verify] {msg}", flush=True)


# ── CPUCache patch (PR #3202) ──────────────────────────────────────────────────
# DistributedCPUCache creates one /dev/shm file per tensor and two collective
# ops per tensor. For 256-expert MoE models this means O(layers × 256 × 3)
# shm files, exhausting physical RAM at 122B scale. CPUCache keeps tensors in
# regular pinned CPU memory (no /dev/shm). The patch is applied to
# LinearExperts2D.from_experts_module so it runs after the fused 3D tensors
# are already broadcast to all ranks — CPUCache.from_mapping then receives
# real tensors on every rank, not meta tensors.
def _cpu_offload_module(module: torch.nn.Module, execution_device: str) -> None:
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
    """Build LinearExperts2D with CPUCache from the already-broadcast 3D tensors."""
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


if not USE_DIST_CACHE:
    LinearExperts2D.from_experts_module = _ddp_from_experts_module
    _log0("Patch: LinearExperts2D → CPUCache (no /dev/shm)")
else:
    _log0(
        "USE_DIST_CACHE=1: CPUCache patch skipped. "
        "DistributedCPUCache will create O(layers × experts × 3) shm files. "
        "WARNING: /dev/shm exhaustion likely on large MoE models "
        "(122B: ~229 GB measured)."
    )


# ── Bug reproduction: broadcast without CPUCache writeback ─────────────────────
def _buggy_broadcast_no_writeback(
    module_list, module_to_rank, qparam_names, skip_cpu=True
):
    """Upstream state before PR #3066: broadcast without CPUCache writeback."""
    pending_comms = []
    for module in module_list:
        should_broadcast = dist.is_initialized() and (
            not skip_cpu or get_execution_device(module) != torch.device("cpu")
        )
        if should_broadcast:
            src = module_to_rank[module]
            for name in qparam_names:
                if (param := getattr(module, name, None)) is not None:
                    pending_comms.append(
                        dist.broadcast(as_broadcastable(param), src=src, async_op=True)
                    )
        obs = getattr(module, "weight_observer", None)
        if obs is not None and obs.has_statistics:
            obs.delete_statistics(check_fused=True)
    wait_for_comms(pending_comms)
    # Intentionally omits update_offload_parameter — this is the bug.


if USE_ORIG_BROADCAST:
    _dist_mod.broadcast_qparams_and_cleanup = _buggy_broadcast_no_writeback
    _gptq_mod.broadcast_qparams_and_cleanup = _buggy_broadcast_no_writeback
    _log0("USE_ORIG_BROADCAST=1: buggy broadcast active → expect CORRUPTED")
else:
    _log0("USE_ORIG_BROADCAST=0: PR #3066 fix active → expect CLEAN")


# ── Weight scale check ─────────────────────────────────────────────────────────
def _check_weight_scale(model: torch.nn.Module) -> None:
    total = nan_c = inf_c = 0
    examples = []
    for name, module in model.named_modules():
        ws = None
        if "weight_scale" in getattr(module, "_parameters", {}):
            ws = module._parameters["weight_scale"]
        elif hasattr(module, "weight_scale"):
            ws = getattr(module, "weight_scale", None)
        if not isinstance(ws, torch.Tensor):
            continue
        total += 1
        try:
            wc = ws.detach().float().cpu()
        except Exception:
            nan_c += 1
            continue
        nan_n = int(torch.isnan(wc).sum())
        inf_n = int(torch.isinf(wc).sum())
        if nan_n:
            nan_c += 1
            if len(examples) < 3:
                examples.append(f"NaN {name}: {nan_n}/{wc.numel()}")
        elif inf_n:
            inf_c += 1
            if len(examples) < 3:
                examples.append(f"Inf {name}: {inf_n}/{wc.numel()}")
    status = "CLEAN" if (nan_c + inf_c) == 0 else "CORRUPTED"
    _log(
        f"[SCALE_CHECK] {status}  total={total} nan={nan_c} inf={inf_c}"
        f"  USE_ORIG_BROADCAST={USE_ORIG_BROADCAST}"
    )
    for ex in examples:
        _log(f"  {ex}")


# ── Offline calibration dataset ───────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

_TEXTS = [
    "The mixture-of-experts architecture routes each token to a sparse subset of "
    "expert feed-forward layers, enabling massive capacity with sub-linear compute.",
    "Large language models are pretrained on vast corpora of text data to learn "
    "general language representations before fine-tuning on specific tasks.",
    "Quantization reduces model precision from float16 to int4, shrinking memory by 4× "
    "while preserving accuracy through careful calibration of weight scales.",
    "Transformer self-attention scales quadratically with sequence length, motivating "
    "efficient attention approximations such as linear attention and sliding window.",
    "Post-training quantization applies calibration data to compute optimal scale and "
    "zero-point parameters without retraining the full model from scratch.",
    "The GPTQ algorithm uses second-order Hessian information to minimise quantization "
    "error column-by-column within each linear layer of the model.",
    "Distributed training parallelises computation across multiple GPUs using data "
    "parallelism, tensor parallelism, or pipeline parallelism strategies.",
    "Expert routing in MoE models is learned jointly with the model weights through a "
    "differentiable gating network that produces sparse token-to-expert assignments.",
    "CPU offloading allows models larger than GPU memory to be calibrated by keeping "
    "inactive layers in host RAM and streaming them on demand.",
    "Weight-only quantization compresses stored weights to low precision while "
    "performing all arithmetic in higher precision for activation compatibility.",
] * (NUM_CALIBRATION_SAMPLES // 10 + 1)

_tokens = [
    tokenizer(t, truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors=None)
    for t in _TEXTS[:NUM_CALIBRATION_SAMPLES]
]
_calib_dataset = Dataset.from_list(_tokens)


# ── Main ───────────────────────────────────────────────────────────────────────
_log0(f"model={MODEL_ID}  N={NUM_CALIBRATION_SAMPLES}  world_size={world_size}")
_log0(f"USE_ORIG_BROADCAST={USE_ORIG_BROADCAST}  USE_DIST_CACHE={USE_DIST_CACHE}")

with load_context(Qwen3_5MoeForConditionalGeneration):
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        torch_dtype="auto",
        trust_remote_code=True,
    )
_log0("model loaded")
dist.barrier()

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
        actorder=None,
    )
]

oneshot(
    model=model,
    dataset=_calib_dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
_log0("GPTQ oneshot complete")
dist.barrier()

_check_weight_scale(model)
dist.barrier()

dist.destroy_process_group()
