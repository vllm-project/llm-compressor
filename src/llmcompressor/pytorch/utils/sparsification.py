"""
Helper functions for retrieving information related to model sparsification
"""

import json
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from accelerate.accelerator import get_state_dict_offloaded_model
from loguru import logger
from torch.nn import Module
from tqdm import tqdm

from llmcompressor.pytorch.utils.helpers import get_quantized_layers, tensor_sparsity

__all__ = [
    "ModuleSparsificationInfo",
    "GradSampler",
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
        return sum(
            torch.numel(self.trainable_params[f"{name}.weight"])
            + (
                torch.numel(self.trainable_params[f"{name}.bias"])
                if hasattr(layer, "bias") and layer.bias is not None
                else 0
            )
            for (name, layer) in get_quantized_layers(self.module)
        )

    @property
    def params_quantized_percent(self) -> float:
        """
        :return: percentage of parameters that have been quantized
        """
        return self.params_quantized / float(self.params_total) * 100


class GradSampler:
    """
    Class for computing gradient samples for a Model given a sample data loader and
    loss function.

    :param data_loader: iterator of data samples to use as model inputs and their loss
        targets. items must be tuples of
        (forward_args: List, forward_kwargs: Dict, loss_targets: Any)
        where the forward pass will be outputs = model(*forward_args, **forward_kwargs)
        and loss will be loss = loss_fn(outputs, loss_targets)
    :param loss_fn: function to be called on model outputs to compute the loss at
        each step
    """

    def __init__(
        self,
        data_loader: Union[Iterator[Tuple[List[Any], Dict[str, Any], Any]], Callable],
        loss_fn: Callable[[Any, Any], Any],
    ):
        if not isinstance(data_loader, Iterable) and not callable(data_loader):
            raise ValueError(
                "data_loader for GradSampler must be Iterable or Callable, received "
                f"object of type {type(data_loader)}"
            )
        if not callable(loss_fn):
            raise ValueError(
                "loss_fn for GradSampler must be callable, given input "
                f"with type {type(loss_fn)}"
            )

        self._data_loader = data_loader
        self._loss_fn = loss_fn

    def iter_module_backwards(
        self,
        module: Module,
        num_grads: int,
        progress_bar: bool = True,
    ) -> Generator[int, None, None]:
        """
        :param module: module to compute gradients for
        :param num_grads: number of gradient samples to compute
        :return: generator that yields after every gradient is computed with the index
            of the gradient sample number
        """
        computed_grads = 0
        pbar = tqdm(
            total=num_grads, desc="Collecting gradients", disable=not progress_bar
        )

        with pbar:
            while computed_grads < num_grads:
                data_loader = (
                    self._data_loader()
                    if callable(self._data_loader)
                    else self._data_loader
                )
                for forward_args, forward_kwargs, loss_target in data_loader:
                    module.zero_grad()
                    # run sample forward and backwards pass
                    model_outputs = module(*forward_args, **forward_kwargs)
                    # Image classification models have been overridden to compute both
                    # the logit values and the probabilities, returning a tuple.
                    #  No other models do this.
                    if model_outputs.__class__ == tuple:
                        model_outputs = model_outputs[0]
                    loss = self._loss_fn(model_outputs, loss_target)
                    loss.backward()

                    # yield so gradients can be collected
                    computed_grads += 1
                    yield computed_grads
                    if progress_bar:
                        pbar.update(1)
                    if computed_grads >= num_grads:
                        break
                if computed_grads < num_grads:
                    logger.warning(
                        f"The requested num_grads:{num_grads} "
                        f"is greater than allowed by the dataset. \
                        Proceeding with less than requested. \
                        Please reduce num_grads to suppress the warning."
                    )
                    break
        module.zero_grad()
