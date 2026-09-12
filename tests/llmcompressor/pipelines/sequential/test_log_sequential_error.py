import math
import re

from loguru import logger

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL = "nm-testing/tinysmokellama-3.2"
NUM_SAMPLES = 8
MAX_SEQ_LENGTH = 128

# e.g. "subgraph 2/7 | sequential error (SQNR dB): 84.91" (or "... : inf")
_SQNR_LINE_PATTERN = re.compile(
    r"subgraph (\d+)/(\d+) \| sequential error \(SQNR dB\): (\S+)"
)


def _run_oneshot_and_capture_metrics(log_sequential_error: bool) -> list[str]:
    """
    Run GPTQ oneshot calibration and capture all METRIC-level log messages.

    Uses a temporary loguru sink (rather than capsys/capfd) since
    `llmcompressor` binds its console sink to `sys.stdout` at import time,
    before any per-test stdout capturing fixture is installed.
    """
    messages = []
    handler_id = logger.add(messages.append, level="METRIC", format="{message}")
    try:
        oneshot(
            model=MODEL,
            dataset="open_platypus",
            splits=f"train[:{NUM_SAMPLES}]",
            # W8A8 (per-channel weights) avoids a group_size divisibility
            # error: this model's hidden_size=64 isn't divisible by W4A16's
            # default group_size of 128
            recipe=GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            pipeline="sequential",
            log_sequential_error=log_sequential_error,
        )
    finally:
        logger.remove(handler_id)

    return messages


def _parse_sqnr_lines(messages: list[str]) -> list[tuple[int, int, float]]:
    """Extract (subgraph_index, num_subgraphs, sqnr_value) from METRIC lines."""
    parsed = []
    for message in messages:
        match = _SQNR_LINE_PATTERN.search(message)
        if match:
            subgraph_index, num_subgraphs, sqnr_value = match.groups()
            parsed.append((int(subgraph_index), int(num_subgraphs), float(sqnr_value)))
    return parsed


def test_log_sequential_error_reports_sqnr():
    """
    Enabling log_sequential_error should log one SQNR value per subgraph,
    except the last (which has no downstream consumer to compare against).
    """
    messages = _run_oneshot_and_capture_metrics(log_sequential_error=True)
    sqnr_lines = _parse_sqnr_lines(messages)
    assert sqnr_lines, "Expected at least one SQNR METRIC line"

    num_subgraphs = sqnr_lines[0][1]
    reported_indices = {subgraph_index for subgraph_index, _, _ in sqnr_lines}
    assert reported_indices == set(range(1, num_subgraphs)), (
        f"Expected SQNR for subgraphs 1..{num_subgraphs - 1}, "
        f"got {sorted(reported_indices)}"
    )

    # SQNR has no lower bound (unlike KL divergence): a negative value is
    # valid if compression noise exceeds the signal. Only NaN is a bug.
    for subgraph_index, _, sqnr in sqnr_lines:
        assert not math.isnan(sqnr), f"subgraph {subgraph_index} SQNR is NaN"


def test_log_sequential_error_disabled_by_default():
    """
    With log_sequential_error at its default (False), no SQNR METRIC lines
    should be emitted.
    """
    messages = _run_oneshot_and_capture_metrics(log_sequential_error=False)
    assert not _parse_sqnr_lines(messages), "Expected no SQNR METRIC lines"
