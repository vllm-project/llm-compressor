from types import SimpleNamespace

import torch
from compressed_tensors.offload import set_onload_device
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import (
    subgraph_offload_modules,
    subgraph_onload_modules,
)

from llmcompressor.modeling.module_restructure import RESTRUCTURE_METADATA_ATTR
from llmcompressor.modeling.offset_norm import CalibrationOffsetNorm
from llmcompressor.pipelines.sequential.helpers import (
    maybe_add_parent_restructure_targets,
)
from llmcompressor.pipelines.sequential.restructure import (
    onload_and_restructure_modules,
    restore_restructured_modules,
)
from llmcompressor.utils.pytorch.meta import copy_module_to_meta


def test_expert_subgraph_materializes_parent_and_offloads_siblings():
    original = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    expected_weights = [module.weight.detach().clone() for module in original]
    target = copy_module_to_meta(original)
    setattr(
        target,
        RESTRUCTURE_METADATA_ATTR,
        {
            "original": original,
            "transform": SimpleNamespace(restructure=lambda source, context: source),
            "context": None,
            "fqn": "experts",
            "materialized": False,
        },
    )
    model = torch.nn.Module()
    model.experts = target
    model.other = torch.nn.Linear(4, 4)
    modules = {"experts.0": target[0], "other": model.other}
    device = torch.device("cpu")
    set_onload_device(original, device)

    assert maybe_add_parent_restructure_targets(model, modules) is modules
    assert modules == {
        "experts.0": target[0],
        "other": model.other,
        "experts": target,
    }
    offload_kwargs = subgraph_onload_modules(modules)
    offload_kwargs.update(onload_and_restructure_modules(modules, device))

    assert modules["experts"] is target
    assert modules["experts.0"] is target[0]
    assert not hasattr(target, RESTRUCTURE_METADATA_ATTR)
    for module, expected in zip(target, expected_weights):
        torch.testing.assert_close(module.weight, expected)

    subgraph_offload_modules(modules, offload_kwargs)
    assert all(isinstance(module._parameters, OffloadCache) for module in target)

    # A later expert loads independently and does not repeat the parent restructure.
    next_modules = {"experts.1": target[1]}
    maybe_add_parent_restructure_targets(model, next_modules)
    assert list(next_modules) == ["experts.1"]
    next_offload_kwargs = subgraph_onload_modules(next_modules)
    assert onload_and_restructure_modules(next_modules, device) == {}
    torch.testing.assert_close(target[1].weight, expected_weights[1])
    subgraph_offload_modules(next_modules, next_offload_kwargs)


def test_norm_restore_updates_same_mapping_before_offload():
    original = torch.nn.Module()
    original.eps = 1e-6
    original.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    expected_weight = original.weight.detach().clone() + 2
    transform = SimpleNamespace(
        restructure=lambda source, context: CalibrationOffsetNorm(source, None)
    )
    target = transform.restructure(copy_module_to_meta(original), None)
    setattr(
        target,
        RESTRUCTURE_METADATA_ATTR,
        {
            "original": original,
            "transform": transform,
            "context": None,
            "fqn": "norm",
            "materialized": False,
        },
    )
    model = torch.nn.Module()
    model.norm = target
    model.other = torch.nn.Linear(4, 4)
    modules = {"norm": target, "other": model.other}
    device = torch.device("cpu")
    set_onload_device(original, device)

    offload_kwargs = subgraph_onload_modules(modules)
    offload_kwargs.update(onload_and_restructure_modules(modules, device))
    with torch.no_grad():
        target.weight.add_(2)

    restore_restructured_modules(model, modules, offload_sources=False)

    assert model.norm is original
    assert modules["norm"] is original
    assert modules["other"] is model.other
    assert not hasattr(target, RESTRUCTURE_METADATA_ATTR)
    torch.testing.assert_close(original.weight, expected_weight)

    subgraph_offload_modules(modules, offload_kwargs)
    assert isinstance(original._parameters, OffloadCache)
    assert not isinstance(target._parameters, OffloadCache)
