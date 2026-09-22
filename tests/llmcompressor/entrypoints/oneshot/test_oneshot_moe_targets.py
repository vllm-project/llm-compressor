import importlib
from types import SimpleNamespace

import torch

oneshot_module = importlib.import_module("llmcompressor.entrypoints.oneshot")
linearize_module = importlib.import_module("llmcompressor.modeling.moe.linearize")


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
        linearize_module,
        "get_moe_linear_status",
        lambda _model: {model.moe: "moe"},
    )

    assert oneshot_module._has_individual_expert_targets(model, ["moe.experts.0"])
    assert not oneshot_module._has_individual_expert_targets(model, ["moe"])


def test_individual_expert_targets_force_non_eager_linearization(monkeypatch):
    model = _Model()
    monkeypatch.setattr(
        linearize_module,
        "get_moe_linear_status",
        lambda _model: {model.moe: "moe"},
    )
    dataset_args = SimpleNamespace(
        sequential_targets=["moe.experts.0"],
        moe_eager_linearization_and_repack=True,
    )

    oneshot_module._force_non_eager_moe_linearization(model, dataset_args)

    assert not dataset_args.moe_eager_linearization_and_repack
