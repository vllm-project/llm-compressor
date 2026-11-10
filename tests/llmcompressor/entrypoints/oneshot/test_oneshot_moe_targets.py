import importlib
from types import SimpleNamespace

import pytest
import torch

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
        "get_moe_modules",
        lambda _model: {model.moe: "moe"},
    )

    assert entrypoint_utils.has_individual_expert_targets(model, ["moe.experts.0"])
    assert not entrypoint_utils.has_individual_expert_targets(model, ["moe"])


def test_individual_expert_targets_force_eager_linearization(monkeypatch):
    model = _Model()
    monkeypatch.setattr(
        entrypoint_utils,
        "get_moe_modules",
        lambda _model: {model.moe: "moe"},
    )
    dataset_args = SimpleNamespace(
        sequential_targets=["moe.experts.0"],
        moe_lazy_linearization_and_repack=True,
    )

    monkeypatch.setattr(
        entrypoint_utils,
        "has_individual_expert_targets",
        lambda _model, _targets: True,
    )
    entrypoint_utils.resolve_eager_moe_linearization(model, dataset_args)

    assert not dataset_args.moe_lazy_linearization_and_repack


@pytest.mark.parametrize(
    ("pipeline", "lazy", "should_process"),
    [
        ("independent", True, False),
        ("sequential", True, False),
        ("sequential", False, True),
        ("basic", True, True),
    ],
)
def test_pre_process_linearizes_moe_when_not_lazy_sequential(
    monkeypatch, pipeline, lazy, should_process
):
    model = _Model()
    model_args = SimpleNamespace(
        model=model,
        processor=object(),
        tie_word_embeddings=True,
    )
    dataset_args = SimpleNamespace(
        sequential_targets=[],
        moe_lazy_linearization_and_repack=lazy,
        pipeline=pipeline,
    )
    calls = []

    monkeypatch.setattr(entrypoint_utils, "is_distributed", lambda: False)
    monkeypatch.setattr(
        entrypoint_utils,
        "linearize_moe",
        lambda candidate: calls.append(candidate),
    )
    monkeypatch.setattr(entrypoint_utils, "modify_save_pretrained", lambda _model: None)

    entrypoint_utils.pre_process(model_args, dataset_args, output_dir=None)

    assert calls == ([model] if should_process else [])


@pytest.mark.parametrize(
    ("pipeline", "lazy", "should_process"),
    [
        ("independent", True, False),
        ("sequential", True, False),
        ("sequential", False, True),
        ("basic", True, True),
    ],
)
def test_post_process_repacks_moe_when_not_lazy_sequential(
    monkeypatch, pipeline, lazy, should_process
):
    model = _Model()
    model_args = SimpleNamespace(model=model, processor=None)
    dataset_args = SimpleNamespace(
        moe_lazy_linearization_and_repack=lazy,
        repack_moe_layers=True,
        pipeline=pipeline,
    )
    calls = []

    monkeypatch.setattr(
        entrypoint_utils,
        "repack_moe",
        lambda candidate: calls.append(candidate),
    )

    entrypoint_utils.post_process(model_args=model_args, dataset_args=dataset_args)

    assert calls == ([model] if should_process else [])
