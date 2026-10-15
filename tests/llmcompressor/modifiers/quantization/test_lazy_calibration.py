import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.quant_config import QuantizationStatus

from llmcompressor.core import (
    Event,
    EventType,
    PipelineInitializationContext,
    State,
)
from llmcompressor.modifiers.quantization import QuantizationModifier


def test_quantization_calibration_initializes_only_scoped_modules():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    args = QuantizationArgs(
        num_bits=8,
        type="int",
        symmetric=True,
        strategy="tensor",
        dynamic=False,
    )
    config = QuantizationScheme(
        targets=["Linear"], weights=args, input_activations=args
    )
    modifier = QuantizationModifier(config_groups={"group": config})
    state = State(model=model)
    modifier.initialize_pipeline(PipelineInitializationContext(state, model))
    event = Event(type_=EventType.CALIBRATION_START)

    modifier.on_calibration_start(state, event, modules=[model[0]])

    assert hasattr(model[0], "input_observer")
    assert not hasattr(model[1], "input_observer")
    assert model[0].quantization_status == QuantizationStatus.CALIBRATION
    assert model[1].quantization_status == QuantizationStatus.INITIALIZED

    first_observer = model[0].input_observer
    modifier.on_calibration_start(state, event, modules=[model[1]])

    assert model[0].input_observer is first_observer
    assert hasattr(model[1], "input_observer")
