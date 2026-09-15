"""
Multi-GPU test for distributed GPTQ.

This test drives ``GPTQModifier`` directly (without ``oneshot``) so that the
distributed compression path in ``GPTQModifier.compress_modules`` is exercised
in isolation.

It is a regression test for the distributed GPTQ hang caused by writing back an
improperly formatted ``weight_global_scale``. When the written tensor does not
match the shape/dtype of the registered offload parameter, ``update_offload_parameter``
falls back to synchronously re-offloading a brand new parameter instead of
updating the existing one in place. In a distributed run only the module's
owning rank performs that write, so the ranks end up disagreeing on the
parameter's shape/dtype, which stalls the collectives that later broadcast and
save these params -- the run hangs.

Because the failure mode is a hang, the process group is initialized with a low
timeout and blocking wait so that any residual deadlock fails fast instead of
hanging the whole test suite. The cross-rank inconsistency is also asserted
directly, so the regression is caught deterministically even when NCCL happens
not to deadlock on the (single-element) parameter.
"""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from compressed_tensors.offload import align_module_device, set_onload_device
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.core import State
from llmcompressor.modifiers.gptq import GPTQModifier
from tests.testing_utils import requires_gpu, torchrun

HIDDEN = 64
NUM_LAYERS = 4
# Keep the timeout short: without the source fix the collectives deadlock, and we
# want a regression to surface as a fast failure rather than a hung test suite.
DIST_TIMEOUT = datetime.timedelta(seconds=60)


class _ToyModel(nn.Module):
    def __init__(self, hidden: int = HIDDEN, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(num_layers)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _init_dist_low_timeout() -> torch.device:
    # Blocking wait makes a stuck collective raise once ``DIST_TIMEOUT`` elapses
    # instead of hanging forever. Must be set before the communicator is created.
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=DIST_TIMEOUT)

    return torch.device(f"cuda:{local_rank}")


@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_gptq_distributed_global_scale_writeback():
    device = _init_dist_low_timeout()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # identical initial weights on every rank
    torch.manual_seed(0)
    model = _ToyModel()

    # Offload params to CPU with this rank's GPU as the onload device. This is
    # what exercises the offload-cache write-back path that the fix targets; a
    # mis-shaped ``weight_global_scale`` triggers a synchronous re-offload here.
    set_onload_device(model, device)

    # tensor_group float weights produce a ``weight_global_scale``, which is the
    # parameter whose write-back formatting caused the distributed hang.
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            strategy="tensor_group",
            group_size=16,
        ),
    )
    modifier = GPTQModifier(config_groups={"group_0": scheme}, block_size=16)

    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)

    # each rank calibrates on different data, as in a real DDP calibration run
    torch.manual_seed(100 + rank)
    modules = [module for module in model.modules() if isinstance(module, nn.Linear)]
    for _ in range(4):
        model(torch.randn(2, 8, HIDDEN, device=device))

    # drive the distributed compression path directly (no oneshot)
    modifier.on_sequential_epoch_end(state, None, modules)

    # every module across every rank must be compressed; nothing left pending
    assert len(modifier._num_samples) == 0
    assert len(modifier._hessians) == 0

    # Each module is compressed on exactly one (owning) rank, which writes back
    # its ``weight_global_scale``. Without the source fix, ``update_offload_parameter``
    # sees a value whose shape/dtype (0-dim, weight dtype) does not match the
    # registered ``[1]`` float32 offload parameter and re-offloads a brand new
    # parameter synchronously instead of updating the existing one in place. That
    # leaves the owning rank with a mis-shaped parameter while every other rank
    # keeps the original ``[1]`` float32 one -- an inconsistency that stalls the
    # collectives which broadcast/save these params across ranks. This assertion
    # (run before any collective, so it fails fast rather than deadlocking a
    # peer) catches that inconsistency directly; the low process-group timeout
    # above is the backstop that turns any residual hang into a fast failure.
    for module in modules:
        assert module.weight_global_scale.shape == torch.Size([1])
        assert module.weight_global_scale.dtype == torch.float32

    # qparams computed on the owning rank are broadcast to all ranks, so the
    # quantized weights must be identical everywhere
    for module in modules:
        with align_module_device(module):
            weight = module.weight.detach().to(device=device, dtype=torch.float32)
        gathered = [torch.empty_like(weight) for _ in range(world_size)]
        dist.all_gather(gathered, weight)
        for other in gathered[1:]:
            assert torch.equal(gathered[0], other)

    dist.destroy_process_group()
