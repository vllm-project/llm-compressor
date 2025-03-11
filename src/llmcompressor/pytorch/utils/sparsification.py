"""
Helper functions for retrieving information related to model sparsification
"""

import json
from typing import Dict, Optional

import torch
from accelerate.accelerator import get_state_dict_offloaded_model
from torch.nn import Module
from tqdm import tqdm

from llmcompressor.pytorch.utils.helpers import get_quantized_layers, tensor_sparsity

__all__ = [
    "ModuleSparsificationInfo",
]


class ModuleSparsificationInfo:
    """
    Helper class for providing information related to torch Module parameters
    and the amount of sparsification applied. Includes information for pruning
    and quantization

    :param module: torch Module to analyze
    :param state_dict: optional state_dict to analyze in place of the torch model. This
    is used when analyzing an FSDP model, where the full weights may not be accessible
    """

    def __init__(
        self, module: Module, state_dict: Optional[Dict[str, torch.Tensor]] = None
    ):
        self.module = module

        if state_dict is not None:
            # when analyzing an FSDP model, the state_dict does not differentiate
            # between trainable and non-trainable parameters
            # (e.g. it can contain buffers) this means that the
            # self.trainable_parameters may be overestimated
            self.trainable_params = state_dict
        else:
            if hasattr(module, "_hf_hook"):
                self.trainable_params = get_state_dict_offloaded_model(module)
            else:
                self.trainable_params = {
                    k: v for k, v in self.module.named_parameters() if v.requires_grad
                }

    def __str__(self):
        return json.dumps(
            {
                "params_summary": {
                    "total": self.params_total,
                    "sparse": self.params_sparse,
                    "sparsity_percent": self.params_sparse_percent,
                    "quantized": self.params_quantized,
                    "quantized_percent": self.params_quantized_percent,
                },
                "params_info": self.params_info,
            }
        )

    @property
    def params_total(self) -> int:
        """
        :return: total number of trainable parameters in the model
        """
        return sum(torch.numel(param) for param in self.trainable_params.values())

    @property
    def params_sparse(self) -> int:
        """
        :return: total number of sparse (0) trainable parameters in the model
        """
        return sum(
            round(tensor_sparsity(param).item() * torch.numel(param))
            for param in tqdm(
                self.trainable_params.values(), desc="Calculating model sparsity"
            )
        )

    @property
    def params_sparse_percent(self) -> float:
        """
        :return: percent of sparsified parameters in the entire model
        """
        return self.params_sparse / float(self.params_total) * 100

    @property
    def params_quantized(self) -> int:
        """
        :return: number of parameters across quantized layers
        """
        num_params = 0
        for name, layer in get_quantized_layers(self.module):
            if getattr(layer, "weight", None) is not None:
                num_params += torch.numel(layer.weight)
            if getattr(layer, "bias", None) is not None:
                num_params += torch.numel(layer.bias)

        return num_params

    @property
    def params_quantized_percent(self) -> float:
        """
        :return: percentage of parameters that have been quantized
        """
        return self.params_quantized / float(self.params_total) * 100
