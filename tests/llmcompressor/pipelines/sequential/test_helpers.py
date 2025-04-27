import pytest
import torch
from torch.fx import _symbolic_trace

from llmcompressor.pipelines.sequential.helpers import (
    add_autowrap_methods,
    get_sequential_ancestors,
)
from llmcompressor.utils.helpers import patch_attr


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU())
        self.fc = torch.nn.Linear(20, 5)

    def forward(self, x):
        x = self.seq(x)
        return self.fc(x)


@pytest.fixture(scope="module")
def model():
    return DummyModel()


def test_get_sequential_ancestors(model):
    assert get_sequential_ancestors(model, set()) == set()
    assert get_sequential_ancestors(model, {model}) == set()
    assert get_sequential_ancestors(model, {model.fc}) == {model}
    assert get_sequential_ancestors(model, {model.seq[0]}) == {model, model.seq}
    assert get_sequential_ancestors(model, {model.seq[1]}) == {model, model.seq}


def test_add_autowrap_methods(model):
    with patch_attr(_symbolic_trace, "_wrapped_methods_to_patch", []):
        add_autowrap_methods(model, ["ReLU.forward"])
        assert _get_matched_modules() == {torch.nn.ReLU}

    with patch_attr(_symbolic_trace, "_wrapped_methods_to_patch", []):
        add_autowrap_methods(model, ["Linear.forward"])
        assert _get_matched_modules() == {torch.nn.Linear}

    with patch_attr(_symbolic_trace, "_wrapped_methods_to_patch", []):
        add_autowrap_methods(model, ["pop"])
        assert _get_matched_modules() == {torch.nn.Sequential}

    with patch_attr(_symbolic_trace, "_wrapped_methods_to_patch", []):
        add_autowrap_methods(model, ["forward"])
        assert _get_matched_modules() == {
            DummyModel,
            torch.nn.Sequential,
            torch.nn.Linear,
            torch.nn.ReLU,
        }


def _get_matched_modules():
    return set(module for module, _ in _symbolic_trace._wrapped_methods_to_patch)
