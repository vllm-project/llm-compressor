import pytest

from llmcompressor.core import Event
from llmcompressor.modifiers.pruning.helpers import (
    PruningCreateSettings,
    cubic_scheduler,
    polynomial_decay_scheduler,
)


def _settings(**kwargs):
    defaults = dict(
        start=0, end=10, update=1, init_sparsity=0.0, final_sparsity=0.9, args={}
    )
    defaults.update(kwargs)
    return PruningCreateSettings(**defaults)


def _run(scheduler, indices):
    values = []
    for i in indices:
        event = Event()
        event.current_index = i
        values.append(scheduler(event, None))
    return values


@pytest.mark.parametrize("exponent", [2, 3, 4])
def test_polynomial_decay_is_monotonic_within_bounds(exponent):
    # AGP decay must rise monotonically from init_sparsity to final_sparsity and
    # stay within that range for even and odd exponents alike. Before the fix,
    # even exponents (including the default of 2) started above final_sparsity
    # and decreased.
    sparsities = _run(
        polynomial_decay_scheduler(_settings(args={"exponent": exponent})),
        range(0, 11),
    )
    assert sparsities[0] == pytest.approx(0.0)
    assert sparsities[-1] == pytest.approx(0.9)
    assert all(0.0 <= s <= 0.9 + 1e-9 for s in sparsities)
    assert all(a <= b + 1e-9 for a, b in zip(sparsities, sparsities[1:]))


def test_polynomial_decay_default_exponent_starts_at_init():
    # The default exponent is 2 (even); before the fix index 0 gave 1.8.
    sched = polynomial_decay_scheduler(_settings())
    assert _run(sched, [0, 10]) == pytest.approx([0.0, 0.9])


def test_cubic_scheduler_unchanged():
    # cubic uses exponent=3 (odd), which was already correct and must not change.
    sched = cubic_scheduler(_settings())
    assert _run(sched, [0, 5, 10]) == pytest.approx([0.0, 0.7875, 0.9])
