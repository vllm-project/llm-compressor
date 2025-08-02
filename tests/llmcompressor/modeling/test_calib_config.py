from unittest.mock import MagicMock

import pytest

from llmcompressor.modeling.prepare import replace_modules_for_calibration


def test_calib_config():
    model = MagicMock()
    with pytest.raises(NotImplementedError) as exc_info:
        replace_modules_for_calibration(model, False, False)

    assert str(exc_info.value) == (
        "At least one of moe_calibrate_gated_acts or "
        "moe_calibrate_all_experts must be set to True."
    )
