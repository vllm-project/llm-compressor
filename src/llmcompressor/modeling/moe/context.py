import contextlib

_CALIBRATE_ALL_EXPERTS = False


def get_calibrate_all_experts_flag() -> bool:
    return _CALIBRATE_ALL_EXPERTS


@contextlib.contextmanager
def moe_calibration_context():
    global _CALIBRATE_ALL_EXPERTS

    restore_value, _CALIBRATE_ALL_EXPERTS = _CALIBRATE_ALL_EXPERTS, True
    try:
        yield
    finally:
        _CALIBRATE_ALL_EXPERTS = restore_value
