import math
import re

from loguru import logger

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL = "nm-testing/tinysmokellama-3.2"
NUM_SAMPLES = 8
MAX_SEQ_LENGTH = 128

# Matches the METRIC message logged by SequentialPipeline once per subgraph when
# log_sequential_error=True, e.g. (as captured via `format="{message}"`, which
# strips the leading timestamp/function/level fields loguru normally adds):
#
#   subgraph 2/7 | sequential error (KL): 0.022716
#
# The first group is the 1-indexed subgraph number, the second is the total
# number of subgraphs, and the third is the mean KL divergence for that subgraph.
_KL_LINE_PATTERN = re.compile(
    r"subgraph (\d+)/(\d+) \| sequential error \(KL\): (-?[\d.eE+-]+)"
)


def _run_oneshot_and_capture_metrics(log_sequential_error: bool) -> list[str]:
    """
    Run GPTQ oneshot calibration on the smoke model and capture all METRIC-level
    log messages emitted during the run.

    A temporary loguru sink is used (rather than capsys/capfd) since
    `llmcompressor` binds its console sink to `sys.stdout` once at import
    time, before any per-test stdout capturing fixture is installed.

    Example return value (tinysmokellama-3.2 has 6 decoder layers, so 7
    subgraphs total; the last subgraph has no downstream consumer and is
    therefore never logged). Values are near-zero since this smoke model's
    randomly-initialized weights barely change under W8A8 quantization; note
    subgraph 5 below, whose value is a small negative floating point rounding
    artifact rather than a true (impossible) negative KL divergence:

        [
            "subgraph 1/7 | sequential error (KL): 0.000000",
            "subgraph 2/7 | sequential error (KL): 0.000001",
            "subgraph 3/7 | sequential error (KL): 0.000000",
            "subgraph 4/7 | sequential error (KL): 0.000002",
            "subgraph 5/7 | sequential error (KL): -0.000001",
            "subgraph 6/7 | sequential error (KL): 0.000000",
        ]
    """
    messages = []
    handler_id = logger.add(messages.append, level="METRIC", format="{message}")
    try:
        oneshot(
            model=MODEL,
            dataset="open_platypus",
            splits=f"train[:{NUM_SAMPLES}]",
            # W8A8 uses per-channel weights (no group_size). tinysmokellama-3.2 has
            # hidden_size=64, which is not divisible by the default W4A16 group_size
            # (128) and would raise a RuntimeError in the weight observer.
            recipe=GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            pipeline="sequential",
            log_sequential_error=log_sequential_error,
        )
    finally:
        logger.remove(handler_id)

    return messages


def _parse_kl_lines(messages: list[str]) -> list[tuple[int, int, float]]:
    """
    Extract (subgraph_index, num_subgraphs, kl_value) tuples from METRIC lines.

    Example:
        >>> _parse_kl_lines(["subgraph 2/7 | sequential error (KL): 0.022716"])
        [(2, 7, 0.022716)]
    """
    parsed = []
    for message in messages:
        match = _KL_LINE_PATTERN.search(message)
        if match:
            subgraph_index, num_subgraphs, kl_value = match.groups()
            parsed.append((int(subgraph_index), int(num_subgraphs), float(kl_value)))
    return parsed


def test_log_sequential_error_reports_kl_divergence():
    """
    Enabling log_sequential_error (see #3154) should emit one KL divergence
    METRIC line per subgraph, except for the last subgraph (which has no
    downstream consumer and is therefore skipped). Every reported value
    should be finite and non-negative, consistent with the definition of KL
    divergence.
    """
    messages = _run_oneshot_and_capture_metrics(log_sequential_error=True)
    kl_lines = _parse_kl_lines(messages)

    assert kl_lines, "Expected at least one 'sequential error (KL)' METRIC line"

    # every line should report the same subgraph total
    subgraph_totals = {num_subgraphs for _, num_subgraphs, _ in kl_lines}
    assert (
        len(subgraph_totals) == 1
    ), f"Expected a single consistent subgraph total, got {subgraph_totals}"
    num_subgraphs = subgraph_totals.pop()
    reported_indices = {subgraph_index for subgraph_index, _, _ in kl_lines}

    # every subgraph except the last should report a KL value
    assert reported_indices == set(range(1, num_subgraphs)), (
        f"Expected KL divergence reported for subgraphs 1..{num_subgraphs - 1}, "
        f"got {sorted(reported_indices)}"
    )

    # KL divergence is mathematically non-negative, but floating point rounding
    # in log_softmax/kl_div can produce values that are negative by a tiny amount
    # when the compared tensors are (near) identical
    floating_point_tolerance = -1e-4
    for subgraph_index, _, kl_value in kl_lines:
        assert math.isfinite(
            kl_value
        ), f"subgraph {subgraph_index} KL divergence is not finite: {kl_value}"
        assert (
            kl_value >= floating_point_tolerance
        ), f"subgraph {subgraph_index} KL divergence is negative: {kl_value}"


def test_log_sequential_error_disabled_by_default():
    """
    With log_sequential_error left at its default (False), no 'sequential error
    (KL)' METRIC lines should be emitted, and calibration should otherwise
    complete normally.
    """
    messages = _run_oneshot_and_capture_metrics(log_sequential_error=False)
    kl_lines = _parse_kl_lines(messages)

    assert not kl_lines, (
        "Expected no 'sequential error (KL)' METRIC lines when "
        f"log_sequential_error is disabled, got {len(kl_lines)}"
    )
