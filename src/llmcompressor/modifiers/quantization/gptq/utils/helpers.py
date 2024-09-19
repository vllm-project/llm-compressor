from typing import Any, Iterable, List, Tuple, Union

import torch

__all__ = ["get_output_error"]


def get_output_error(
    unquantized: List[Tuple[Union[Iterable, torch.Tensor], Any]],
    quantized: List[Tuple[Union[Iterable, torch.Tensor], Any]],
) -> torch.Tensor:
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

    assert len(unquantized_outputs) == len(quantized_outputs)
    return sum(
        [
            torch.nn.functional.l1_loss(unq, q)
            for unq, q in zip(unquantized_outputs, quantized_outputs)
        ]
    )
