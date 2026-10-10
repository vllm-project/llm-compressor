"""
Multi-GPU regression test for distributed GPTQ with GPU offloading (#2480).

Drives ``GPTQModifier`` directly (without ``oneshot``) so the distributed
compression path is exercised in isolation. Each rank holds the toy model with
every layer offloaded to the *peer* GPU (``DeviceCache``/``DistributedDeviceCache``
with ``onload_device != offload_device``). After GPTQ quantization and the
qparam broadcast, both the onloaded *and* the offloaded copies must agree across
ranks; before the #2480 fix the offloaded (peer-GPU) copies on non-owning ranks
were never updated.

Requires 2 GPUs; skipped on smaller hosts.
"""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from compressed_tensors.offload import offload_module
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.core import State
from llmcompressor.modifiers.gptq import GPTQModifier
from tests.testing_utils import requires_gpu, torchrun

HIDDEN = 64


class _ToyModel(nn.Module):
    def __init__(self, hidden=HIDDEN, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(num_layers)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_gptq_distributed_gpu_offload_broadcast():
    # blocking wait so a stuck collective raises on timeout instead of hanging
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
    try:
        _run_offload_check(local_rank)
    finally:
        # always tear down the process group so a failed assert cannot hang the
        # next test / the runner on a dangling NCCL state
        dist.destroy_process_group()


def _run_offload_check(local_rank: int):
    device = torch.device(f"cuda:{local_rank}")
    peer_device = torch.device(f"cuda:{1 - local_rank}")

    torch.manual_seed(0)
    model = _ToyModel()

    # GPU offload: weights live on the peer GPU, execute on this rank's GPU
    for layer in model.layers:
        offload_module(layer, onload_device=device, offload_device=peer_device)

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

    torch.manual_seed(100 + dist.get_rank())  # different calibration data per rank
    modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for _ in range(4):
        model(torch.randn(2, 8, HIDDEN, device=device))

    modifier.on_sequential_epoch_end(state, None, modules)

    world_size = dist.get_world_size()
    for module in modules:
        cache = module._parameters

        # no module may have been left on CPU: the whole point of the feature
        assert str(cache.offload_device).startswith("cuda")

        # offloaded copies (peer GPU) must agree across ranks after broadcast
        offloaded = (
            cache.offloaded_values["weight"]
            .detach()
            .to(device=device, dtype=torch.float32)
        )
        gathered_offload = [torch.empty_like(offloaded) for _ in range(world_size)]
        dist.all_gather(gathered_offload, offloaded)
        assert all(
            torch.equal(gathered_offload[0], other) for other in gathered_offload[1:]
        )

        # onloaded copies must agree across ranks as well
        onloaded = (
            cache.onload(cache.offloaded_values["weight"])
            .detach()
            .to(device=device, dtype=torch.float32)
        )
        gathered_onload = [torch.empty_like(onloaded) for _ in range(world_size)]
        dist.all_gather(gathered_onload, onloaded)
        assert all(
            torch.equal(gathered_onload[0], other) for other in gathered_onload[1:]
        )
