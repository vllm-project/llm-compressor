# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.quantization import (
    initialize_module_for_quantization,
    preset_name_to_scheme,
)

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.observers import Observer


def _observed_qparams(scheme_name: str):
    scheme = preset_name_to_scheme(scheme_name, ["Linear"])
    observer = Observer.load_from_registry(
        "minmax",
        base_name="weight",
        args=scheme.weights,
    )
    observed = torch.arange(1, 33, dtype=torch.float32).reshape(1, 32)
    return observer(observed).get_qparams()


def test_no_global_scale_observer_returns_no_global_scale():
    qparams = _observed_qparams("NVFP4_E5M3")

    assert qparams["global_scale"] is None
    assert torch.isfinite(qparams["scale"]).all()
    assert qparams["scale"].max() <= 114688.0


def test_ue5m3_global_scale_uses_extended_range():
    qparams = _observed_qparams("NVFP4_E5M3_GS")

    expected_global_scale = 114688.0 * 6.0 / 32.0
    assert torch.isclose(
        qparams["global_scale"],
        torch.tensor([expected_global_scale]),
    ).all()
    assert qparams["scale"].max() == 114688.0


def test_no_gs_requires_no_calibration_and_attaches_no_input_observer():
    modifier = QuantizationModifier(targets="Linear", scheme="NVFP4_E5M3")
    assert modifier.requires_calibration_data is False

    scheme = preset_name_to_scheme("NVFP4_E5M3", ["Linear"])
    module = torch.nn.Linear(32, 32)
    initialize_module_for_quantization(module, scheme)
    module.quantization_scheme = scheme
    modifier._initialize_observers(module)

    assert hasattr(module, "weight_observer")
    assert not hasattr(module, "input_observer")


def test_gs_requires_calibration():
    modifier = QuantizationModifier(targets="Linear", scheme="NVFP4_E5M3_GS")
    assert modifier.requires_calibration_data is True
