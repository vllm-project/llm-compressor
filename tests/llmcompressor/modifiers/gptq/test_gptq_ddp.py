"""
Multi-GPU regression test for distributed GPTQ.

Drives ``GPTQModifier`` directly (without ``oneshot``) so the distributed
compression path is exercised in isolation. Without the source fix, the owning
rank writes back a mis-shaped ``weight_global_scale``, leaving ranks disagreeing
on its shape/dtype and hanging the collectives that broadcast/save it. The
process group uses a low timeout so any residual deadlock fails fast.
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
def test_gptq_distributed_global_scale_writeback():
    # blocking wait so a stuck collective raises on timeout instead of hanging
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(0)
    model = _ToyModel()
    set_onload_device(model, device)  # CPU offload, GPU onload

    # tensor_group float weights are the case that produces weight_global_scale
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, type="float", symmetric=True,
            strategy="tensor_group", group_size=16,
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

    # weight_global_scale must be consistently shaped across ranks, asserted
    # before any collective so a regression fails fast rather than deadlocking
    for module in modules:
        assert module.weight_global_scale.shape == torch.Size([1])
        assert module.weight_global_scale.dtype == torch.float32

    # quantized weights are broadcast, so they must be identical on every rank
    world_size = dist.get_world_size()
    for module in modules:
        with align_module_device(module):
            weight = module.weight.detach().to(device=device, dtype=torch.float32)
        gathered = [torch.empty_like(weight) for _ in range(world_size)]
        dist.all_gather(gathered, weight)
        assert all(torch.equal(gathered[0], other) for other in gathered[1:])

    dist.destroy_process_group()
