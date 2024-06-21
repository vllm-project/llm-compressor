from contextlib import nullcontext as does_not_raise

import pytest

from llmcompressor.metrics.utils import FrequencyManager


@pytest.fixture(scope="function")
def frequency_manager():
    return FrequencyManager(log_frequency=1)


@pytest.mark.parametrize(
    "kwargs, expectation",
    [
        (
            {"log_frequency": 0},
            pytest.raises(ValueError, match="must be greater than 0, given 0"),
        ),
        (
            {"log_frequency": -1},
            pytest.raises(ValueError, match="must be greater than 0, given -1"),
        ),
        (
            {"log_frequency": True},
            pytest.raises(TypeError, match="must be a number or None"),
        ),
        (
            {"log_frequency": []},
            pytest.raises(TypeError, match="must be a number or None"),
        ),
        (
            {"log_frequency": {}},
            pytest.raises(TypeError, match="must be a number or None"),
        ),
        ({"log_frequency": 1}, does_not_raise()),
        ({"log_frequency": None}, does_not_raise()),
    ],
)
def test_frequency_manager_creation(kwargs, expectation):
    with expectation:
        _ = FrequencyManager(**kwargs)


@pytest.mark.parametrize(
    "step, expectation",
    [
        (0.1, does_not_raise()),
        (0, does_not_raise()),
        (-1, pytest.raises(ValueError, match="must be greater than or equal to 0")),
        (True, pytest.raises(TypeError, match="must be a number or None")),
        ([], pytest.raises(TypeError, match="must be a number or None")),
        ({}, pytest.raises(TypeError, match="must be a number or None")),
    ],
)
class TestFrequencyManagerUpdationUtilities:
    def test_model_updated(self, frequency_manager, step, expectation):
        # test that model_updated sets last_model_update_step
        # to the given step

        with expectation:
            frequency_manager.model_updated(step=step)
            assert frequency_manager.last_model_update_step == step

    def test_log_written(self, frequency_manager, step, expectation):
        # test that log_written sets last_log_step
        # to the given step

        with expectation:
            frequency_manager.log_written(step=step)
            assert frequency_manager.last_log_step == step


def _log_ready_test_cases():
    # test cases for log_ready

    # each test case is a tuple of:
    #   (log_frequency, current_log_step, last_log_step,
    #       last_model_update_step, check_model_update, expected)

    return [
        # None values should give True
        (0.1, None, None, None, False, True),
        (0.1, None, 1, 1, False, True),
        (0.1, 1, None, 1, False, True),
        (0.1, 0.3, 0.2, None, False, True),
        (0.1, 0.3, 0.2, None, True, True),
        # log frequency is None
        (None, 1, 2, 3, False, False),
        (None, 1, 2, 3, True, False),
        # cadence not reached
        (0.1, 1, 1, 0.1, False, False),
        (0.1, 1, 1, 0.1, True, False),
        # cadence reached
        (0.1, 0.3, 0.1, 0.3, False, True),
        (0.1, 0.3, 0.1, 0.1, True, True),
        # model updated long back and
        # and cadence reached
        (0.1, 0.3, 0.1, 0.1, True, True),
    ]


@pytest.mark.parametrize(
    "log_frequency, current_log_step, last_log_step,"
    " last_model_update_step, check_model_update, expected",
    _log_ready_test_cases(),
)
def test_log_ready(
    log_frequency,
    current_log_step,
    last_log_step,
    last_model_update_step,
    check_model_update,
    expected,
):
    frequency_manager = FrequencyManager(log_frequency=log_frequency)
    frequency_manager.last_log_step = last_log_step
    frequency_manager.last_model_update_step = last_model_update_step

    actual = frequency_manager.log_ready(
        current_log_step=current_log_step, check_model_update=check_model_update
    )

    assert actual == expected


@pytest.mark.parametrize(
    "log_frequency, frequency_type, expectation",
    [
        # epoch frequency type accepts floats
        (0.1, "epoch", does_not_raise()),
        # batch and step frequency types need
        # log_frequency to be an integer
        (0.1, "step", pytest.raises(TypeError)),
        # negative log_frequency is invalid
        (-1, "epoch", pytest.raises(ValueError)),
        (-1, "step", pytest.raises(ValueError)),
    ],
)
def test__validate_log_frequency(log_frequency, frequency_type, expectation):
    with expectation:
        FrequencyManager(log_frequency=log_frequency, frequency_type=frequency_type)
