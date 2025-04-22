import pytest
import torch.nn as nn

from llmcompressor.utils.pytorch import get_parent_by_name


@pytest.fixture
def example_nested_module() -> str:
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.Sequential(nn.ReLU(), nn.Linear(20, 10)),
        nn.Sequential(nn.SiLU(), nn.Linear(20, 10)),
        nn.Softmax(dim=1),
    )


@pytest.mark.unit
def test_get_parent_by_name(example_nested_module):
    # Test getting the parent of the first layer
    name, parent = get_parent_by_name("0", example_nested_module)
    assert parent == example_nested_module

    # Test getting the parent of a nested layer
    name, parent = get_parent_by_name("1.0", example_nested_module)
    assert parent == example_nested_module[1]
    assert name == "1"

    name, parent = get_parent_by_name("1.1", example_nested_module)
    assert parent == example_nested_module[1]
    assert name == "1"

    name, parent = get_parent_by_name("2.0", example_nested_module)
    assert parent == example_nested_module[2]
    assert name == "2"

    name, parent = get_parent_by_name("2.1", example_nested_module)
    assert parent == example_nested_module[2]
    assert name == "2"

    # Test getting the parent of a non-existent layer
    with pytest.raises(ValueError):
        get_parent_by_name("non_existent_layer", example_nested_module)
