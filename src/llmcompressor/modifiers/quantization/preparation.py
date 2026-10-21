"""CPU-only allocation jobs. Target resolution and installation stay on the caller."""

from dataclasses import dataclass, field

import torch
from compressed_tensors.quantization import QuantizationScheme, create_qparams

from llmcompressor.modifiers.quantization.calibration import create_observer
from llmcompressor.observers import Observer


@dataclass
class PreparedQuantization:
    parameters: dict[torch.nn.Module, dict[str, torch.nn.Parameter]] = field(
        default_factory=dict
    )
    observers: dict[torch.nn.Module, dict[str, Observer]] = field(default_factory=dict)


@dataclass
class QuantizationPreparation:
    targets: list[
        tuple[torch.nn.Module, QuantizationScheme, tuple[int, ...], torch.dtype]
    ]
    with_observers: bool = True

    def __call__(self) -> PreparedQuantization:
        result = PreparedQuantization()
        for module, scheme, shape, dtype in self.targets:
            parameters = {}
            observers = {}
            for name, args, observed_shape in (
                ("input", scheme.input_activations, shape[-1:]),
                ("weight", scheme.weights, shape),
                ("output", scheme.output_activations, shape[:-1]),
            ):
                if args is None:
                    continue
                parameters.update(create_qparams(name, args, observed_shape, dtype))
                if self.with_observers and not (name == "output" and args.dynamic):
                    observer = create_observer(args, name)
                    if observer is not None:
                        observers[name] = observer
            result.parameters[module] = parameters
            result.observers[module] = observers
        return result
