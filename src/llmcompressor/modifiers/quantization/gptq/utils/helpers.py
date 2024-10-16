from typing import Any, Iterable, List, Tuple, Union

import torch

__all__ = ["get_output_error", "gptq_hook"]


def get_output_error(
    unquantized: List[Tuple[Union[Iterable, torch.Tensor], Any]],
    quantized: List[Tuple[Union[Iterable, torch.Tensor], Any]],
) -> torch.Tensor:
    """
    Calculate mean l1 loss between weight-unquantized outputs and weight-quantized
    outputs

    :param unquantized: unquantized-weight outputs
    :param quantized: quantized-weight outputs
    :return: mean l1 loss between outputs
    """
    unquantized_outputs = sum(
        [
            [output for output in outputs]
            if isinstance(outputs, Iterable)
            else [outputs]
            for outputs, _ in unquantized
        ],
        start=[],
    )

    quantized_outputs = sum(
        [
            [output for output in outputs]
            if isinstance(outputs, Iterable)
            else [outputs]
            for outputs, _ in quantized
        ],
        start=[],
    )

    if len(unquantized_outputs) != len(quantized_outputs):
        raise ValueError(
            "Number of samples of weight-unquantized and weight-quantized "
            "outputs differs"
        )

    return sum(
        [
            torch.nn.functional.l1_loss(unq, q)
            for unq, q in zip(unquantized_outputs, quantized_outputs)
        ]
    ) / len(unquantized_outputs)


def gptq_hook(func):
    def wrapped(self, *args, **kwargs):
        if self.hooks_disabled:
            return
        
        func(self, *args, **kwargs)

    return wrapped