"""Integration test: real CPUCache + real dist.broadcast.

Demonstrates the writeback bug and its fix in a single distributed session.

Key design:
- offload_module is called BEFORE dist.init_process_group().
  OffloadCache.cls_from_device checks dist.is_initialized() at call time.
  When dist is not yet initialized, offload_device='cpu' selects CPUCache
  (not DistributedCPUCache), so each rank holds independent CPU storage.
- dist is initialized AFTER module setup, enabling real NCCL broadcast.

This mirrors the real auto_offload DDP GPTQ scenario:
  - GPTQ runs on each module's owning rank, computes weight_scale
  - broadcast_qparams_and_cleanup is called to propagate to all other ranks
  - Without the fix, dist.broadcast modifies a temporary CUDA tensor that is
    discarded without writing back to each rank's independent CPUCache storage
"""

import os

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.distributed import wait_for_comms
from compressed_tensors.offload import offload_module
from compressed_tensors.offload.dist_utils import as_broadcastable

from llmcompressor.utils.dist import broadcast_qparams_and_cleanup
from tests.testing_utils import requires_gpu, torchrun


def _upstream_broadcast_no_writeback(module_list, module_to_rank, qparam_names):
    """Reproduces upstream (buggy) broadcast_qparams_and_cleanup:
    broadcasts without writing back to CPUCache storage."""
    pending_comms = []
    for module in module_list:
        src = module_to_rank[module]
        for name in qparam_names:
            if (param := getattr(module, name, None)) is not None:
                pending_comms.append(
                    dist.broadcast(as_broadcastable(param), src=src, async_op=True)
                )
    wait_for_comms(pending_comms)
    # Intentionally no update_offload_parameter — this is the bug


@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2, init_dist=False)
def test_broadcast_qparams_writeback_with_cpu_offload():
    """Phase A: upstream bug — non-src rank keeps stale weight_scale after broadcast.
    Phase B: fix — all ranks get correct weight_scale after broadcast + writeback.

    Critical: offload_module is called BEFORE dist.init_process_group() so that
    OffloadCache.cls_from_device('cpu') returns CPUCache (not DistributedCPUCache),
    giving each rank its own independent CPU memory for weight_scale.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)

    src_val = 42.0
    stale_val = 0.0

    # ── Setup: create modules with independent CPUCache BEFORE dist init ─────
    # At this point dist.is_initialized() == False → offload_module selects CPUCache
    def _make_module(val: float) -> torch.nn.Module:
        mod = torch.nn.Linear(4, 4, bias=False).to(device)
        offload_module(mod, onload_device=device, offload_device="cpu")
        mod.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.full((4,), val if rank == 0 else stale_val),
                requires_grad=False,
            ),
        )
        return mod

    mod_bug = _make_module(src_val)  # used for Phase A (upstream bug demo)
    mod_fix = _make_module(src_val)  # used for Phase B (fix demo)
    module_to_rank = {mod_bug: 0, mod_fix: 0}

    # ── Init dist AFTER module setup ─────────────────────────────────────────
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=int(os.environ["WORLD_SIZE"]),
        device_id=device,
    )

    # ── Phase A: upstream (buggy) broadcast — no CPUCache writeback ───────────
    _upstream_broadcast_no_writeback([mod_bug], module_to_rank, ["weight_scale"])
    dist.barrier()

    result_a = getattr(mod_bug, "weight_scale").cpu()
    if rank == 0:
        assert torch.all(
            result_a == src_val
        ), f"rank 0 (src): expected {src_val}, got {result_a}"
    else:
        # Bug: dist.broadcast wrote into a temporary CUDA tensor which was discarded.
        # CPUCache storage was never updated → non-src rank still holds stale_val.
        assert torch.all(result_a == stale_val), (
            f"rank {rank}: upstream bug — expected stale {stale_val} in CPUCache, "
            f"got {result_a} (bug not present or test setup incorrect)"
        )
        print(
            f"\n[rank {rank}] Phase A BUG CONFIRMED: "
            f"weight_scale={result_a.tolist()} (stale, not {src_val})"
        )

    # ── Phase B: fix — broadcast_qparams_and_cleanup with writeback ───────────
    broadcast_qparams_and_cleanup([mod_fix], module_to_rank, ["weight_scale"])
    dist.barrier()

    result_b = getattr(mod_fix, "weight_scale").cpu()
    assert torch.all(result_b == src_val), (
        f"rank {rank}: Phase B (fix) — expected {src_val} after writeback, "
        f"got {result_b}"
    )
    print(f"[rank {rank}] Phase B FIX CONFIRMED: weight_scale={result_b.tolist()}")

    dist.destroy_process_group()
