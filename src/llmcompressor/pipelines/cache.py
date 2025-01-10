import warnings
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import tqdm


@dataclass
class IntermediateValue:
    """
    Dataclass which recursively defines offloaded values and which device to onload to

    :param value: either an offloaded Tensor, an primative value, or a recursable value
    :param device: if the value is a Tensor, then the device to onload the tensor to,
        otherwise None
    """

    value: Union[torch.Tensor, "IntermediateValue", Any]
    device: Union[torch.device, None]


class IntermediatesCache:
    """
    Cache which stores intermediate values (activations) produced by batched, sequential
    execution of models. Values are offloaded to the `offload_device` when stored in
    the cache and onloaded to their original device when fetched from the cache

    Currently supports nested offloading of dataclass instances and tuples

    Construct using `empty` and `from_dataloader` class methods
    """

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
        """
        Construct an empty cache

        :param num_batches: the expected number of batches to be stored
        :param offload_device: device to offload values to
        """
        batch_intermediates = [{} for _ in range(num_batches)]
        return cls(batch_intermediates, offload_device)

    @classmethod
    def from_dataloader(
        cls,
        dataloader: torch.utils.data.DataLoader,
        model_device: torch.device,
        mask_padding: bool = True,
        offload_device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize a cache with data from the provided dataloader

        :param dataloader: dataloader which generates values to be cached
        :param model_device: device which values will be onloaded to when fetched
        :param mask_padding: zero out padding tokens if True. This affects modifiers
            such as GPTQ and SparseGPT
        :param offload_device: device to offload values to
        """
        # note: list comprehesion was found to not improve performance
        batch_intermediates = []
        for batch in tqdm.tqdm(dataloader, desc="Preparing intermediates cache"):
            intermediate = {}
            for key, value in batch.items():
                if mask_padding and key == "input_ids":
                    value = cls._mask_padding(value, batch["attention_mask"])
                intermediate[key] = IntermediateValue(value=value, device=model_device)

            batch_intermediates.append(intermediate)

        return cls(batch_intermediates, offload_device)

    def fetch(
        self, batch_index: int, input_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch values belonging to a batch

        :param batch_index: index of batch whose values are being fetched
        :param input_names: list of keys whose values are being fetched
        :return: dictionary mapping keys to onloaded values
        """
        intermediates = self.batch_intermediates[batch_index]

        return {
            key: self._onload_value(subgraph_input)
            for key, subgraph_input in intermediates.items()
            if input_names is None or key in input_names
        }

    def update(self, batch_index: int, values: Dict[str, Any]):
        """
        Update/put values belonging to a batch

        :param batch_index: index of batch whose values will be updated
        :param values: dictionary mapping keys to values used for update
        """
        intermediates = {k: self._offload_value(v) for k, v in values.items()}
        self.batch_intermediates[batch_index].update(intermediates)

    def delete(self, batch_index: int, consumed_names: Optional[List[str]] = None):
        """
        Delete values from the cache

        :param batch_index: index of batch whose values will be deleted
        :param consumed_names: list of keys whose values will be deleted, defaults to
            removing all keys
        """
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

        # Assumes that `attention_mask` only contains zeros and ones
        return input_ids * attention_mask
