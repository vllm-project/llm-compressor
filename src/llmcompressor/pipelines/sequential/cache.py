import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import tqdm


@dataclass
class IntermediateValue:
    value: Any
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
                        value=value.masked_fill_(batch["attention_mask"] == 0, 0),
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

    def fetch(self, batch_index: int, input_names: List[str]) -> Dict[str, Any]:
        intermediates = self.batch_intermediates[batch_index]

        return {
            key: self._onload_value(subgraph_input)
            for key, subgraph_input in intermediates.items()
            if key in input_names
        }

    def update(self, batch_index: int, outputs: Dict[str, Any]):
        # assume that all model intermediates are tensors
        assert (isinstance(value, torch.Tensor) for value in outputs.values())

        intermediates = {
            key: self._offload_value(value) for key, value in outputs.items()
        }

        self.batch_intermediates[batch_index].update(intermediates)

    def delete(self, batch_index: int, consumed_names: List[str]):
        intermediates = self.batch_intermediates[batch_index]
        for name in consumed_names:
            del intermediates[name]

    def _onload_value(self, intermediate: IntermediateValue) -> Any:
        value = intermediate.value
        device = intermediate.device

        if device is not None:
            if isinstance(value, torch.Tensor):
                return value.to(device=device)
            else:
                raise NotImplementedError("Intermediates")

        else:
            return value

    def _offload_value(self, value: Any) -> IntermediateValue:
        if isinstance(value, torch.Tensor):
            return IntermediateValue(
                value=value.to(device=self.offload_device), device=value.device
            )

        else:
            warnings.warn(f"Offloading not implemented for type {type(value)}.")
            return IntermediateValue(value=value, device=None)
