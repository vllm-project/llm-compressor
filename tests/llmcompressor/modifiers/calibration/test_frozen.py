from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from torch.nn import Linear

from llmcompressor.modifiers.quantization.calibration import (
    freeze_module_quantization,
    initialize_observer,
)


def test_set_module_for_calibration():
    num_bits = 8
    quantization_scheme = QuantizationScheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=False),
    )

    layer = Linear(4, 4)

    initialize_module_for_quantization(layer, quantization_scheme)
    layer.quantization_status = QuantizationStatus.CALIBRATION
    initialize_observer(layer, "weight")

    # should have both input and weight observer after initalizing
    assert hasattr(layer, "weight_observer")

    # observers should get deleted after freezing
    freeze_module_quantization(layer)
    assert not hasattr(layer, "input_observer")
    assert not hasattr(layer, "weight_observer")

    assert layer.quantization_status == QuantizationStatus.FROZEN
