import time

import torch
import torch.nn as nn
from loguru import logger

from llmcompressor.modifiers.utils.compression_wrapper import ModuleCompressionWrapper

try:
    import transformers
except ImportError as err:
    transformers = None
    transformers_err = err

__all__ = ["WandaWrapper"]


DEBUG = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class WandaWrapper(ModuleCompressionWrapper):
    """
    Runs WANDA on a single module that contains no sub-modules
    see https://arxiv.org/abs/2306.11695

    Runs SparseGPT on a single module that contains no sub-modules

    Lifecycle:
        - add_batch
        - compress
        - free

    :param name: name of module to run compression on
    :param layer: module to run compression on
    """

    def __init__(self, name, layer):
        super().__init__(name=name, layer=layer)
        self.register_buffer("scaler_row", torch.zeros(self.columns, device=self.dev))

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        Add a batch of layer input and output data to the layer statistics calculation

        :param inp: tensor containing layer input
        :param out: tensor containing layer output
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def compress(
        self,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
    ):
        """
        Run pruning on the layer up to the target sparsity value.

        :param sparsity: target sparsity to reach for layer
        :param prunen: N for N:M pruning
        :param prunem: M for N:M pruning
        """
        final_shape = self.layer.weight.shape
        final_dtype = self.layer.weight.dtype
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        W_metric = torch.abs(W) * torch.sqrt(self.scaler_row.reshape((1, -1)))

        # initialize a mask to be all False
        W_mask = torch.zeros_like(W_metric) == 1
        if prunen != 0:
            # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prunem == 0:
                    tmp = W_metric[:, ii : (ii + prunem)].float()
                    W_mask.scatter_(
                        1,
                        ii + torch.topk(tmp, prunen, dim=1, largest=False)[1],
                        True,
                    )
        else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity)]
            W_mask.scatter_(1, indices, True)

        W[W_mask] = 0  # set weights to zero

        logger.info("time %.2f" % (time.time() - tick))

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        W = W.reshape(final_shape).to(final_dtype)

        # This is a bit hacky, but FSDP updates only work if we change the weight in
        # place, clone() or direct assignment won't work
        self.layer.weight -= self.layer.weight
        self.layer.weight += W

    def free(self):
        """
        Free memory after the layer is complete
        """
        delattr(self, "scaler_row")
        super().free()
