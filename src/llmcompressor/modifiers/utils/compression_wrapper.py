from abc import ABC
from typing import Optional, Set

import torch
import torch.nn as nn
from torch.nn import Module

try:
    import transformers
except ImportError as err:
    transformers = None
    transformers_err = err

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

__all__ = ["ModuleCompressionWrapper"]


class ModuleCompressionWrapper(Module, ABC):
    """
    Base Abstract class for pruning/quantization a single module with no sub-modules
    using information from input/output statistics

    Lifecycle:
        - add_batch
        - compress
        - free

    :param name: name of module to run compression on
    :param layer: module to run compression on
    """

    def __init__(self, name, layer):
        super(ModuleCompressionWrapper, self).__init__()
        if transformers is None:
            raise transformers_err

        self.name = name
        self.layer = layer

        self.dev = self.layer.weight.device
        if hasattr(self.layer, "_hf_hook") and self.layer._hf_hook.offload:
            self.dev = self.layer._hf_hook.execution_device

        # Calculate weight shape to use during pruning
        W = self.layer.weight
        if isinstance(self.layer, nn.Conv2d):
            self.rows = W.shape[0]
            self.columns = W.numel() / self.rows
        if isinstance(self.layer, transformers.Conv1D):
            self.rows = W.shape[1]
            self.columns = W.shape[0]
        else:
            self.rows = W.shape[0]
            self.columns = W.shape[1]

        # This need to be a buffer so its preserved between forward passes
        self.register_buffer(
            "nsamples", torch.zeros(1, dtype=torch.int32, device=self.dev)
        )

    def forward(self, *args, **kwargs):
        """
        Run a forward pass of the wrapped layer
        """
        return self.layer(*args, **kwargs)

    def free(self):
        """
        Free buffers used for compression
        """
        delattr(self, "nsamples")

    def add_batch(self, *args, **kwargs):
        """
        Add a batch of layer input and output data to the layer statistics calculation
        """
        raise NotImplementedError("Child class must implement `add_batch`")

    def compress(self, *args, **kwargs):
        """
        Run pruning on the layer up to the target sparsity
        """
        raise NotImplementedError("Child class must implement `compress`")

    def state_dict(self, destination=None, prefix="", keep_vars=False, **kwargs):
        """
        Pass request to wrapped layer, so compression wrapper does not appear in
        the state_dict
        """
        return self.layer.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars, **kwargs
        )

    def load_state_dict(self, state_dict, strict=True):
        """
        Pass request to wrapped layer, so compression wrapper does not appear in
        the state_dict
        """
        return self.layer.load_state_dict(state_dict, strict=strict)

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        """
        Pass request to wrapped layer, so compression wrapper does not appear in
        the module list
        """
        return self.layer.named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )
