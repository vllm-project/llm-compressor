# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline driver data contracts, including real variable-shape sequential runs."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from examples.glm52_precision import oneshot_profile as driver
from llmcompressor.modifiers.transform import FlexSmoothModifier, QuaRotModifier
from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline
from llmcompressor.utils import dev
from tests.flex_smooth.test_distributed_oneshot import _config, _fixture, _run


@pytest.mark.parametrize(
    "world,requested,expected",
    [
        (1, None, "auto"),
        (1, "auto", "auto"),
        (1, "auto_offload", "auto_offload"),
        (2, None, "auto_offload"),
        (2, "auto_offload", "auto_offload"),
    ],
)
def test_placement_contract(world, requested, expected):
    assert driver.resolve_device_map(requested, world) == expected


def test_distributed_auto_rejected():
    with pytest.raises(ValueError, match="auto is single-process only"):
        driver.resolve_device_map("auto", 2)


def _args(path, batch_size=1):
    return SimpleNamespace(
        dataset=str(path), samples=3, sequence_length=16, batch_size=batch_size
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("side", ["left", "right"])
@torch.no_grad()
def test_variable_length_batches_reach_sequential(
    tmp_path, monkeypatch, batch_size, side
):
    _config(tmp_path)
    model, tokenizer = _fixture(tmp_path)
    tokenizer.padding_side = side
    # All data are invented; no external ModelSlim prompts are used.
    texts = ["w1", "w2 w3 w4 w5 w6", "w7 w8 w9"]
    path = tmp_path / "list.json"
    path.write_text(json.dumps(texts))
    report = {}
    driver.model_preflight(model, FlexSmoothModifier(), report)
    loader, _ = driver.calibration_preflight(
        _args(path, batch_size),
        report,
        tokenizer=tokenizer,
        generation_config=model.generation_config,
    )
    expected = [
        tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts
    ]
    actual = []
    for batch in loader:
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
            valid = int(mask.sum())
            assert set(mask.tolist()) <= {0, 1}
            assert (ids[mask == 0] == tokenizer.pad_token_id).all()
            wanted = [1] * valid + [0] * (len(mask) - valid)
            assert mask.tolist() == (wanted if side == "right" else wanted[::-1])
            actual.append(ids[mask.bool()].tolist())
    assert actual == expected
    assert report["samples_local"] == report["samples_global"] == 3
    assert report["tokens_local_valid"] == report["tokens_global_valid"] == 9
    assert report["sequence_length_mean"] == 3
    assert report["sequence_length_min"] == 1
    assert report["sequence_length_max"] == 5
    assert report["batches_local"] == (3 + batch_size - 1) // batch_size
    # A CPU-only host must also catch accidental reliance on auto-detection.
    # Only the pipeline's imported lookup is replaced; global detection stays real.
    host_selector = Mock(side_effect=AssertionError("Host device selection escaped"))
    global_selector = dev.get_main_device
    monkeypatch.setattr(sequential_pipeline, "get_main_device", host_selector)
    with driver.Profile("cpu").instrument():
        diagnostics = _run(model, tokenizer, loader)
    host_selector.assert_not_called()
    assert sequential_pipeline.get_main_device is host_selector
    assert dev.get_main_device is global_selector
    assert len(diagnostics) == 6  # Real three-Modifier sequential lifecycle completed.


@pytest.mark.parametrize("format", ["list", "jsonl", "messages"])
def test_local_formats_preserve_text(tmp_path, format):
    _config(tmp_path)
    model, tokenizer = _fixture(tmp_path)
    texts = ["short", "a somewhat longer sentence", "third calibration prompt"]
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    path = tmp_path / "calibration.json"
    if format == "list":
        path.write_text(json.dumps(texts))
    else:
        rows = [
            {"text": text}
            if format == "jsonl"
            else {"messages": [{"role": "user", "content": text}]}
            for text in texts
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows))
    loader, stats = driver.calibration_data(_args(path, 2), tokenizer)
    assert [row["input_ids"] for row in loader.dataset] == [
        tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts
    ]
    assert stats["tokens_local_valid"] == 8
    assert len(loader) == 2


@pytest.mark.parametrize("cap", [None, 128])
def test_local_flex_cap_preflight(tmp_path, cap):
    _config(tmp_path)
    model, _ = _fixture(tmp_path)
    assert FlexSmoothModifier(max_tokens=cap)._preflight(model)


@pytest.mark.parametrize(
    "failure",
    [
        "malformed",
        "empty",
        "missing_text",
        "empty_tokens",
        "padding",
        "partition",
        "tokenizer",
    ],
)
def test_driver_data_errors_fail_before_quarot(tmp_path, monkeypatch, failure):
    _config(tmp_path)
    model, tokenizer = _fixture(tmp_path)
    path = tmp_path / "bad.json"
    content = json.dumps(["w1", "w2 w3", "w4 w5 w6"])
    if failure == "malformed":
        content = "[not json"
    elif failure == "empty":
        content = "[]"
    elif failure == "missing_text":
        content = '{"other": "w1"}'
    elif failure == "empty_tokens":
        content = '["", "w1", "w2"]'
    elif failure == "padding":
        tokenizer.pad_token = None
    elif failure == "partition":
        content = '["w1"]'
    path.write_text(content)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--model",
            str(tmp_path / "input"),
            "--output",
            str(tmp_path / "out"),
            "--dataset",
            str(path),
            "--samples",
            "3",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
    )
    monkeypatch.setattr(driver, "load_context", nullcontext)
    load_weights = Mock(side_effect=AssertionError("Weights must not load"))
    monkeypatch.setattr(driver.AutoModelForCausalLM, "from_pretrained", load_weights)

    def load_tokenizer(*args, **kwargs):
        if failure == "tokenizer":
            raise ValueError("Invalid local tokenizer")
        return tokenizer

    monkeypatch.setattr(driver.AutoTokenizer, "from_pretrained", load_tokenizer)
    rotate = Mock(side_effect=AssertionError("QuaRot must not start"))
    monkeypatch.setattr(QuaRotModifier, "on_calibration_start", rotate)
    with pytest.raises(ValueError, match="preflight|partition"):
        driver.main()
    rotate.assert_not_called()
    load_weights.assert_not_called()
    report = json.loads((tmp_path / "out-profile/rank-0.json").read_text())
    assert report["status"] == "failed"


def test_cli_requires_explicit_dataset_and_defaults_batch_one(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["profile", "--model", "input", "--output", "out"])
    with pytest.raises(SystemExit):
        driver.parse_args()
    monkeypatch.setattr(sys, "argv", sys.argv + ["--dataset", "local.json"])
    assert driver.parse_args().batch_size == 1
