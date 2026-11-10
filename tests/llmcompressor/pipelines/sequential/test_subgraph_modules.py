import pytest
import torch
from torch.fx import Graph

from llmcompressor.pipelines.sequential.helpers import (
    INVOKED_SUBMODULES_META_KEY,
    Subgraph,
)


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        self.alias = self.block
        self.lookups = []

    def get_submodule(self, target):
        self.lookups.append(target)
        return super().get_submodule(target)


def make_subgraph():
    graph = Graph()
    value = graph.placeholder("x")
    value = graph.call_module("block", args=(value,))
    value.meta[INVOKED_SUBMODULES_META_KEY] = ["alias"]
    graph.output(value)
    return Subgraph(graph, {"x"}, set())


@pytest.mark.unit
@pytest.mark.parametrize("recurse", [False, True])
def test_module_discovery_retains_order_and_shared_aliases(recurse):
    model = CountingModel()
    subgraph = make_subgraph()
    expected = {"block": model.block}
    if recurse:
        expected.update({"block.0": model.block[0], "block.1": model.block[1]})
    assert subgraph.submodule_dict(model, recurse=recurse) == expected
    assert model.lookups == ["block", "alias"]
    model.lookups.clear()
    assert subgraph.submodules(model, recurse=recurse) == list(expected.values())
    assert model.lookups == ["block", "alias"]
    assert subgraph._subgraph_module_names(model, recurse=recurse) == list(expected)


@pytest.mark.unit
def test_module_discovery_observes_replacement_and_graph_changes():
    model = CountingModel()
    subgraph = make_subgraph()
    first = subgraph.submodule_dict(model)
    replacement = torch.nn.Identity()
    model.block[0] = replacement
    assert subgraph.submodule_dict(model)["block.0"] is replacement
    assert first["block.0"] is not replacement
    other_model = CountingModel()
    assert subgraph.submodules(other_model)[0] is other_model.block
    node = next(node for node in subgraph.graph.nodes if node.op == "call_module")
    node.target = "block.0"
    node.meta[INVOKED_SUBMODULES_META_KEY] = []
    assert subgraph.submodule_dict(model) == {"block.0": replacement}
    node.target = "missing"
    with pytest.raises(AttributeError):
        subgraph.submodule_dict(model)
