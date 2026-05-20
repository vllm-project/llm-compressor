import contextlib
from typing import Optional

from transformers import PreTrainedModel

CALIBRATE_ALL_EXPERTS = False


def get_moe_calibration_context() -> bool:
    return CALIBRATE_ALL_EXPERTS


@contextlib.contextmanager
def moe_calibration_context(
    model: Optional[PreTrainedModel] = None, calibrate_all_experts: bool = True
):
    from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D

    global CALIBRATE_ALL_EXPERTS

    # validation
    if model is not None:
        needs_linearization = False
        for module in model.modules():
            if isinstance(module, LinearExperts2D):
                break
            if isinstance(module, FusedExpertsProtocol):
                needs_linearization = True
                break

        if needs_linearization:
            raise ValueError(
                "Attempting to calibrate MoE model without first linearizing. Load "
                "model `with llmcompressor.modeling.moe.linearize.load_quantizable_moe`"
                " before passing into `oneshot`"
            )
            # TODO: in-memory replacement

    restore_value, CALIBRATE_ALL_EXPERTS = CALIBRATE_ALL_EXPERTS, calibrate_all_experts
    try:
        yield
    finally:
        CALIBRATE_ALL_EXPERTS = restore_value
