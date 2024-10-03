import time

from compressed_tensors.quantization.lifecycle.forward import (
    maybe_calibrate_or_quantize,
)

from llmcompressor.modifiers.utils.compression_wrapper import ModuleCompressionWrapper
from llmcompressor.utils import getattr_chain

try:
    import transformers
except ImportError as err:
    transformers = None
    transformers_err = err

import math

import torch
import torch.nn as nn
from loguru import logger

__all__ = ["SparseGptWrapper"]


class SparseGptWrapper(ModuleCompressionWrapper):
    """
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

        # for Hessian calculation
        self.register_buffer(
            "H", torch.zeros((self.columns, self.columns), device=self.dev)
        )

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        Add a batch of layer input and output data to the Hessian calculation

        :param inp: tensor containing layer input
        :param out: tensor containing layer output
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t()).to(self.dev)

    def compress(
        self,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        preserve_sparsity_mask: bool = False,
    ):
        """
        Run pruning and quantization(if applicable) on the layer up to the target
        sparsity value.

        :param sparsity: target sparsity to reach for layer
        :param prunen: N for N:M pruning
        :param prunem: M for N:M pruning
        :param blocksize: Number of columns to compress in one pass
        :param percdamp: Amount of dampening to apply to H, as a fraction of the
            diagonal norm
        :param preserve_sparsity_mask: Extend or ignore the base sparsity mask
        """
        final_shape = self.layer.weight.shape
        final_dtype = self.layer.weight.dtype
        W = self.layer.weight.data.clone()
        args_loc = "quantization_scheme.weights"
        weight_quant_args = getattr_chain(self.layer, args_loc, None)
        if weight_quant_args is not None:
            W = maybe_calibrate_or_quantize(self.layer, W, "weight", weight_quant_args)

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        dead = torch.diag(self.H) == 0
        self.H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(self.H))
        diag = torch.arange(self.columns, device=self.dev)
        self.H[diag, diag] += damp
        self.H = torch.linalg.cholesky(self.H)
        self.H = torch.cholesky_inverse(self.H)
        self.H = torch.linalg.cholesky(self.H, upper=True)
        Hinv = self.H

        mask = None
        if preserve_sparsity_mask:
            # compute existing sparsity mask
            mask = torch.where(
                W == 0,
                torch.tensor(1, dtype=torch.bool),
                torch.tensor(0, dtype=torch.bool),
            )
            current_sparsity = mask.sum() / W.numel()
            if current_sparsity > sparsity:
                raise ValueError(
                    "The target sparsity is lower than the sparsity "
                    "of the base model. Please retry "
                    "after turning preserve_sparsity_mask=False"
                )

        # See section 3.4 of https://arxiv.org/abs/2203.07259
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                    if int(W1.numel() * sparsity) > mask1.sum():
                        # target sparsity is higher than base sparsity, extend mask1
                        tmp = (
                            (~mask[:, i1:i2])
                            * W1**2
                            / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        )
                        thresh = torch.sort(tmp.flatten())[0][
                            int(tmp.numel() * sparsity)
                        ]
                        mask1 = tmp <= thresh
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = (
                        W1[:, i : (i + prunem)] ** 2
                        / (torch.diag(Hinv1)[i : (i + prunem)].reshape((1, -1))) ** 2
                    )
                    if mask is not None:
                        tmp = tmp * (~mask[:, i : (i + prunem)])

                    mask1.scatter_(
                        1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            if preserve_sparsity_mask:
                # respect the sparsity of other groups
                # really not needed, but kept for explicitness
                W[:, i2:] -= (~mask[:, i2:]) * Err1.matmul(Hinv[i1:i2, i2:])
            else:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        logger.info("time %.2f" % (time.time() - tick))
        logger.info("error %.2f" % torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.reshape(final_shape).to(final_dtype)
        if weight_quant_args is not None:
            W = maybe_calibrate_or_quantize(self.layer, W, "weight", weight_quant_args)

        # This is a bit hacky, but FSDP updates only work if we change the weight in
        # place, clone() or direct assignment won't work
        self.layer.weight -= self.layer.weight
        self.layer.weight += W

    def free(self):
        """
        Free the Hessian memory after the layer is complete
        """
        delattr(self, "H")
        super().free()
