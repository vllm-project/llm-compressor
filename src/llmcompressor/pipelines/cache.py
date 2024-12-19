import warnings
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import tqdm


@dataclass
class IntermediateValue:
    value: Union[torch.Tensor, "IntermediateValue", Any]
    device: Union[torch.device, None]


class IntermediatesCache:
    batch_intermediates: List[Dict[str, IntermediateValue]]
    offload_device: torch.device

    def __init__(
        self,
        batch_intermediates: List[Dict[str, IntermediateValue]],
        offload_device: torch.device,
    ):
        self.batch_intermediates = batch_intermediates
        self.offload_device = offload_device

    @classmethod
    def empty(cls, num_batches: int, offload_device: torch.device):
        batch_intermediates = [{} for _ in range(num_batches)]
        return cls(batch_intermediates, offload_device)

    @classmethod
    def from_dataloader(
        cls,
        dataloader: torch.utils.data.DataLoader,
        model_device: torch.device,
        mask_padding: bool = True,
        offload_device: torch.device = "cpu",
    ):
        batch_intermediates = [
            {
                key: (
                    IntermediateValue(
                        value=cls._mask_padding(value, batch["attention_mask"]),
                        device=model_device,
                    )
                    if mask_padding and key == "input_ids"
                    else IntermediateValue(value=value, device=model_device)
                )
                for key, value in batch.items()
            }
            for batch in tqdm.tqdm(dataloader, desc="Preparing intermediates cache")
        ]

        return cls(batch_intermediates, offload_device)

    def fetch(
        self, batch_index: int, input_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        intermediates = self.batch_intermediates[batch_index]

        return {
            key: self._onload_value(subgraph_input)
            for key, subgraph_input in intermediates.items()
            if input_names is None or key in input_names
        }

    def update(self, batch_index: int, values: Dict[str, Any]):
        intermediates = {k: self._offload_value(v) for k, v in values.items()}
        self.batch_intermediates[batch_index].update(intermediates)

    def delete(self, batch_index: int, consumed_names: Optional[List[str]] = None):
        intermediates = self.batch_intermediates[batch_index]

        if consumed_names is None:
            consumed_names = list(intermediates.keys())

        for name in consumed_names:
            del intermediates[name]

    def _onload_value(self, intermediate: IntermediateValue) -> Any:
        value = intermediate.value
        device = intermediate.device

        if isinstance(value, torch.Tensor):
            return value.to(device=device)

        elif is_dataclass(value):
            for field in fields(value):  # `asdict` is recursive, not applicable here
                v = getattr(value, field.name)
                setattr(value, field.name, self._onload_value(v))

            return value

        elif isinstance(value, tuple):
            return tuple(self._onload_value(v) for v in value)

        elif isinstance(value, (int, str, float, bool)) or value is None:
            return value

        else:
            return value

    def _offload_value(self, value: Any) -> IntermediateValue:
        if isinstance(value, torch.Tensor):
            return IntermediateValue(
                value=value.to(device=self.offload_device), device=value.device
            )

        elif is_dataclass(value):
            for field in fields(value):  # `asdict` is recursive, not applicable here
                v = getattr(value, field.name)
                setattr(value, field.name, self._offload_value(v))

            return IntermediateValue(value=value, device=None)

        if isinstance(value, tuple):
            return IntermediateValue(
                value=tuple(self._offload_value(v) for v in value), device=None
            )

        if isinstance(value, (int, str, float, bool)) or value is None:
            return IntermediateValue(value=value, device=None)

        else:
            warnings.warn(f"Offloading not implemented for type {type(value)}.")
            return IntermediateValue(value=value, device=None)

    @staticmethod
    def _mask_padding(
        input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if attention_mask.dim() == 4:
            # some attention masks, such as those from pixtral, are are 4d
            attention_mask = attention_mask[0, 0, 0].unsqueeze(0)

        return input_ids.masked_fill_(torch.logical_not(attention_mask), 0)
