import importlib
from types import SimpleNamespace

import torch

oneshot_module = importlib.import_module("llmcompressor.entrypoints.oneshot")
linearize_module = importlib.import_module("llmcompressor.modeling.moe.linearize")
entrypoint_utils = importlib.import_module("llmcompressor.entrypoints.utils")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = torch.nn.Module()
        self.moe.experts = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
        )


def test_has_individual_expert_targets(monkeypatch):
    model = _Model()
    monkeypatch.setattr(
        entrypoint_utils,
        "get_moe_linear_status",
        lambda _model: {model.moe: "moe"},
    )

    assert oneshot_module.has_individual_expert_targets(model, ["moe.experts.0"])
    assert not oneshot_module.has_individual_expert_targets(model, ["moe"])


def test_individual_expert_targets_force_eager_linearization(monkeypatch):
    model = _Model()
    monkeypatch.setattr(
        entrypoint_utils,
        "get_moe_linear_status",
        lambda _model: {model.moe: "moe"},
    )
    dataset_args = SimpleNamespace(
        sequential_targets=["moe.experts.0"],
        moe_lazy_linearization_and_repack=True,
    )

    oneshot_instance = SimpleNamespace(model=model, dataset_args=dataset_args)
    oneshot_module.Oneshot.resolve_eager_moe_linearization(oneshot_instance)

    assert not dataset_args.moe_lazy_linearization_and_repack
