"""Multi-GPU regression test for distributed GPTQ global-scale synchronization."""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from compressed_tensors.offload import align_module_device, set_onload_device

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
def test_gptq_distributed_global_scale_broadcast():
    # blocking wait so a stuck collective raises on timeout instead of hanging
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(0)
    model = _ToyModel()
    set_onload_device(model, device)  # CPU offload, GPU onload

    modifier = GPTQModifier(scheme="NVFP4A16", targets=["Linear"], block_size=16)

    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)

    torch.manual_seed(100 + dist.get_rank())  # different calibration data per rank
    modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for _ in range(4):
        model(torch.randn(2, 8, HIDDEN, device=device))

    modifier.on_sequential_epoch_end(state, None, modules)

    # Check shape and dtype before gathering values so mismatched parameters fail
    # before entering a tensor collective.
    for module in modules:
        assert module.weight_global_scale.shape == torch.Size([1])
        assert module.weight_global_scale.dtype == torch.float32

    world_size = dist.get_world_size()
    for module in modules:
        with align_module_device(module):
            global_scale = module.weight_global_scale.detach().to(device=device)
        gathered_global_scales = [
            torch.empty_like(global_scale) for _ in range(world_size)
        ]
        dist.all_gather(gathered_global_scales, global_scale)
        assert all(
            torch.equal(gathered_global_scales[0], other)
            for other in gathered_global_scales[1:]
        ), "weight_global_scale differs across ranks"

        # Quantized weights must also be identical on every rank.
        with align_module_device(module):
            weight = module.weight.detach().to(device=device, dtype=torch.float32)
        gathered = [torch.empty_like(weight) for _ in range(world_size)]
        dist.all_gather(gathered, weight)
        assert all(torch.equal(gathered[0], other) for other in gathered[1:])

    dist.destroy_process_group()
