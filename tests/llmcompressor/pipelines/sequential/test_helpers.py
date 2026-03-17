import pytest
import torch

from llmcompressor.pipelines.sequential.helpers import (
    dispatch_for_sequential,
    get_sequential_ancestors,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU())
        self.fc = torch.nn.Linear(20, 5)

    def forward(self, x):
        x = self.seq(x)
        return self.fc(x)


def test_get_sequential_ancestors():
    model = DummyModel()

    assert get_sequential_ancestors(model, set()) == set()
    assert get_sequential_ancestors(model, {model}) == set()
    assert get_sequential_ancestors(model, {model.fc}) == {model}
    assert get_sequential_ancestors(model, {model.seq[0]}) == {model, model.seq}
    assert get_sequential_ancestors(model, {model.seq[1]}) == {model, model.seq}


@pytest.mark.parametrize("offload_device", [None, "none", "None"])
def test_dispatch_for_sequential_no_offload(offload_device):
    """When offload_device is None or 'none', the model should stay on the
    onload device without any offloading hooks."""
    model = DummyModel()
    result = dispatch_for_sequential(
        model, onload_device="cpu", offload_device=offload_device
    )
    assert result is model
    for name, param in model.named_parameters():
        assert param.device == torch.device(
            "cpu"
        ), f"{name} on {param.device}, expected cpu"
