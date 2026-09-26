import math
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
import torch
from loguru import logger
from torch.utils.data import DataLoader

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import active_session
from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline

# known signal and noise for predictable SQNR
SIGNAL = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
NOISE = torch.tensor([[0.1, -0.1, 0.05, -0.05]])
# smaller than SIGNAL (1 element vs 4); never the largest tensor, so it
# should never affect the SQNR below
DECOY = torch.tensor([[99.0]])
NUM_BATCHES = 2
NUM_SUBGRAPHS = 3

# SQNR = 10 * log10(signal_power / noise_power)
#   signal_power = 1^2 + 2^2 + 3^2 + 4^2 = 30
#   noise_power  = 0.1^2 + 0.1^2 + 0.05^2 + 0.05^2 = 0.025
#   SQNR = 10 * log10(30 / 0.025) ≈ 30.79 dB
EXPECTED_SQNR = 10 * math.log10(30.0 / 0.025)

_PIPELINE = "llmcompressor.pipelines.sequential.pipeline"


@dataclass
class FakeSubgraph:
    """
    Returns SIGNAL in pass 1 and SIGNAL+NOISE in pass 2, alongside a
    smaller decoy tensor and a non-tensor value. This exercises
    _get_largest_tensor's "pick the biggest among several, skip
    non-tensors" branch: "h" (4 elements) must be selected over
    "decoy" (1 element) and "note" (not a tensor at all).
    """

    input_names: set[str]
    consumed_names: set[str]
    _call_count: int = field(default=0, repr=False)

    def forward(self, model, **kwargs):
        self._call_count += 1
        h = SIGNAL if self._call_count <= NUM_BATCHES else SIGNAL + NOISE
        return {"h": h, "decoy": DECOY, "note": "not a tensor"}

    def submodules(self, model):
        return []


def _sqnr_values(messages):
    """Extract SQNR floats from captured METRIC log messages."""
    tag = "(SQNR dB): "
    values = []
    for msg in messages:
        pos = msg.find(tag)
        if pos >= 0:
            values.append(float(msg[pos + len(tag) :]))
    return values


@pytest.fixture()
def fake_subgraphs():
    return [
        FakeSubgraph(input_names={"h"}, consumed_names=set())
        for _ in range(NUM_SUBGRAPHS)
    ]


@pytest.fixture()
def fake_dataloader():
    samples = [{"h": SIGNAL} for _ in range(NUM_BATCHES)]
    return DataLoader(samples, batch_size=1, collate_fn=lambda b: b[0])


@pytest.fixture()
def fake_pipeline():
    """Set up model, session, and log capture for pipeline tests."""
    model = torch.nn.Linear(4, 4)

    session = active_session()
    session.reset()
    session.initialize(model=model, start=-1)

    messages = []
    sink = logger.add(messages.append, level="METRIC", format="{message}")

    yield model, messages

    logger.remove(sink)
    session.finalize()


@patch(f"{_PIPELINE}.infer_sequential_targets", return_value=["Linear"])
@patch(f"{_PIPELINE}.trace_subgraphs")
def test_enabled_propagate_true(
    mock_trace, mock_targets, fake_pipeline, fake_subgraphs, fake_dataloader
):
    """
    SQNR = 10 * log10(signal_power / noise_power)
         = 10 * log10(30 / 0.025) ≈ 30.79 dB

    signal_power = 1^2 + 2^2 + 3^2 + 4^2 = 30
    noise_power  = 0.1^2 + 0.1^2 + 0.05^2 + 0.05^2 = 0.025
    """
    mock_trace.return_value = fake_subgraphs
    model, messages = fake_pipeline

    SequentialPipeline()(
        model,
        fake_dataloader,
        DatasetArguments(log_sequential_error=True, propagate_error=True),
    )

    sqnr_values = _sqnr_values(messages)
    assert len(sqnr_values) == NUM_SUBGRAPHS - 1
    assert sqnr_values[0] == pytest.approx(EXPECTED_SQNR, abs=0.01)


@patch(f"{_PIPELINE}.infer_sequential_targets", return_value=["Linear"])
@patch(f"{_PIPELINE}.trace_subgraphs")
def test_enabled_propagate_false(
    mock_trace, mock_targets, fake_pipeline, fake_subgraphs, fake_dataloader
):
    """Same SQNR as test_enabled_propagate_true, via the decoupled
    code path (zero-copy transfer + SQNR)."""
    mock_trace.return_value = fake_subgraphs
    model, messages = fake_pipeline

    SequentialPipeline()(
        model,
        fake_dataloader,
        DatasetArguments(log_sequential_error=True, propagate_error=False),
    )

    sqnr_values = _sqnr_values(messages)
    assert len(sqnr_values) == NUM_SUBGRAPHS - 1
    assert sqnr_values[0] == pytest.approx(EXPECTED_SQNR, abs=0.01)


@patch(f"{_PIPELINE}.infer_sequential_targets", return_value=["Linear"])
@patch(f"{_PIPELINE}.trace_subgraphs")
def test_disabled_propagate_true(
    mock_trace, mock_targets, fake_pipeline, fake_subgraphs, fake_dataloader
):
    """No SQNR when disabled, propagate_error=True."""
    mock_trace.return_value = fake_subgraphs
    model, messages = fake_pipeline

    SequentialPipeline()(
        model,
        fake_dataloader,
        DatasetArguments(log_sequential_error=False, propagate_error=True),
    )

    assert _sqnr_values(messages) == []


@patch(f"{_PIPELINE}.infer_sequential_targets", return_value=["Linear"])
@patch(f"{_PIPELINE}.trace_subgraphs")
def test_disabled_propagate_false(
    mock_trace, mock_targets, fake_pipeline, fake_subgraphs, fake_dataloader
):
    """No SQNR when disabled, propagate_error=False."""
    mock_trace.return_value = fake_subgraphs
    model, messages = fake_pipeline

    SequentialPipeline()(
        model,
        fake_dataloader,
        DatasetArguments(log_sequential_error=False, propagate_error=False),
    )

    assert _sqnr_values(messages) == []
