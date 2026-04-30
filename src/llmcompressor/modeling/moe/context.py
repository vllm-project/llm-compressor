import contextlib

from transformers import PreTrainedModel

from . import linearize

CALIBRATE_ALL_EXPERTS = False


@contextlib.contextmanager
def moe_calibration_context(model: PreTrainedModel, calibrate_all_experts: bool):
    global CALIBRATE_ALL_EXPERTS

    linearize.linearize_moe_model(model)

    restore_value = CALIBRATE_ALL_EXPERTS
    CALIBRATE_ALL_EXPERTS = calibrate_all_experts
    try:
        yield
    finally:
        CALIBRATE_ALL_EXPERTS = restore_value
