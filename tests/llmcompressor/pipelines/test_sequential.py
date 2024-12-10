from functools import partial
from unittest.mock import Mock

import pytest
import torch
import torch.utils.data.dataloader
from accelerate import dispatch_model
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from llmcompressor.pipelines.sequential import run_pipeline
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs


class LinearWithBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.bias = torch.nn.Parameter(
            torch.tensor(
                1.0,
            )
        )

    def forward(self, x):
        return self.linear(x) + self.bias


class ParallelLinears(torch.nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.parallel0 = LinearWithBias()
        self.parallel1 = LinearWithBias()
        self.bias = torch.nn.Parameter(
            torch.tensor(
                1.0,
            )
        )

    def forward(self, x):
        return self.parallel0(x) + self.parallel1(x) + self.bias


class SequentialLinears(torch.nn.Module):
    """
    SequentialLinears(
        (sequential0): ParallelLinears(
            (parallel0): LinearWithBias(
                (linear): Linear()
            )
            (parallel1): LinearWithBias(
                (linear): Linear()
            )
        )
        (sequential1): ParallelLinears(
            (parallel0): LinearWithBias(
                (linear): Linear()
            )
            (parallel1): LinearWithBias(
                (linear): Linear()
            )
        )
    """

    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.sequential0 = ParallelLinears()
        self.sequential1 = ParallelLinears()

        self.config = Mock(use_cache=True)
        self.device = device

    @property
    def dummy_input(self):
        return {"x": torch.tensor([1.0])}

    def forward(self, x):
        return self.sequential1(self.sequential0(x))


def patched(original_fn, *args, **kwargs):
    # accelerate.dispatch_model sometimes gets the type of the tensor, then uses this
    # value to instantiate a new tensor. The signature for Tensor.__new__ and
    # FakeTensor.__new__ are different, so this patch catches that case
    if len(args) >= 2 and isinstance(args[1], FakeTensor):
        return args[1]

    return original_fn(*args, **kwargs)


@pytest.mark.parametrize(
    "device_map",
    [
        {"": "cpu"},
        {"": "cuda:0"},
        {"sequential0": "cuda:0", "sequential1": "cuda:1"},
    ],
)
def test_trace_subgraphs(device_map, monkeypatch):
    with FakeTensorMode(), monkeypatch.context() as m:
        m.setattr(FakeTensor, "__new__", partial(patched, FakeTensor.__new__))
        model = SequentialLinears()
        dispatch_model(model, device_map)

        targets = [m for m in model.modules() if type(m).__name__ == "Linear"]
        subgraphs = trace_subgraphs(model, model.dummy_input, targets)
        assert len(subgraphs) == len(targets) + 1


@pytest.mark.parametrize(
    "device_map",
    [
        {"": "cpu"},
        {"": "cuda:0"},
        {"sequential0": "cuda:0", "sequential1": "cuda:1"},
    ],
)
def test_run_pipeline(device_map, monkeypatch):
    with FakeTensorMode(), monkeypatch.context() as m:
        m.setattr(FakeTensor, "__new__", partial(patched, FakeTensor.__new__))
        model = SequentialLinears()
        dispatch_model(model, device_map)

        dataloader = (model.dummy_input, model.dummy_input)
        run_pipeline(model, ["Linear"], [], dataloader, propagate_error=True)
