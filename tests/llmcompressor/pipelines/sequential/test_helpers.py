import torch
import torch.fx
from llmcompressor.pipelines.sequential.helpers import get_sequential_ancestors
from llmcompressor.pipelines.sequential.helpers import topological_partition
import pytest



class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU())
        self.fc = torch.nn.Linear(20, 5)

    def forward(self, x):
        x = self.seq(x)
        return self.fc(x)

class DummyModelMultipleSequentialLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)
        self.layer3 = torch.nn.Linear(10, 10)
        self.layer4 = torch.nn.Linear(10, 10)
        self.layer5 = torch.nn.Linear(10, 10)
        self.layer6 = torch.nn.Linear(10, 10)
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
        
def test_get_sequential_ancestors():
    model = DummyModel()

    assert get_sequential_ancestors(model, set()) == set()
    assert get_sequential_ancestors(model, {model}) == set()
    assert get_sequential_ancestors(model, {model.fc}) == {model}
    assert get_sequential_ancestors(model, {model.seq[0]}) == {model, model.seq}
    assert get_sequential_ancestors(model, {model.seq[1]}) == {model, model.seq}

def test_topological_partition_default():
    model = DummyModelMultipleSequentialLayers()
    targets = {model.layer1, model.layer2, model.layer3, model.layer4, model.layer5, model.layer6}
    gm = torch.fx.symbolic_trace(model)
    
    
    assert len(topological_partition(gm, targets)) == 7

def test_topological_partition_multiple_targets():
    model = DummyModelMultipleSequentialLayers()
    gm = torch.fx.symbolic_trace(model)
    targets = {model.layer1, model.layer2, model.layer3, model.layer4, model.layer5, model.layer6}

    assert len(topological_partition(gm, targets, 2)) == 4

def test_topological_partition_invalid():
    model = DummyModelMultipleSequentialLayers()
    gm = torch.fx.symbolic_trace(model)
    targets = {model.layer1, model.layer2, model.layer3, model.layer4, model.layer5, model.layer6}

    
    with pytest.raises(ValueError):
        topological_partition(gm, targets, 0)

    
    
    
    
    