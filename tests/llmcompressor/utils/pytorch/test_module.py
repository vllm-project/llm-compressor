import pytest
import torch.nn as nn

@pytest.fixture
def example_nested_module() -> str:
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.Sequential(nn.ReLU(), nn.Linear(20, 10)),
        nn.Sequential(nn.SiLU(), nn.Linear(20, 10)),
        nn.Softmax(dim=1),
    )


