import time

from llmcompressor.modifiers.utils import SPARSITY_THRESHOLD
from llmcompressor.modifiers.utils.compression_wrapper import ModuleCompressionWrapper

try:
    import transformers
except ImportError as err:
    transformers = None
    transformers_err = err

import math
from copy import copy

import torch
import torch.nn as nn
from compressed_tensors.utils import (
    get_offloaded_device,
    is_module_offloaded,
    update_prefix_dict,
)
from loguru import logger

__all__ = ["GPTQWrapper"]


class GPTQWrapper(ModuleCompressionWrapper):
    """
    Runs GPTQ on a single module that contains no sub-modules

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
        self.H += inp.matmul(inp.t())

    def compress(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ):
        """
        Run pruning and quantization(if applicable) on the layer up to the target
        sparsity value.

        :param blocksize: Number of columns to compress in one pass
        :param percdamp: Amount of dampening to apply to H, as a fraction of the
            diagonal norm
        """
        if is_module_offloaded(self.layer):
            self.layer._hf_hook.pre_forward(self.layer)

        final_shape = self.layer.weight.shape
        final_dtype = self.layer.weight.dtype
        W = self.layer.weight.data.clone()
        H = self.H.clone()
        from llmcompressor.pytorch.utils.helpers import tensor_sparsity

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        # if activation ordering is enabled, permute the weight columns
        # in order of greatest hessian values. Columns are unpermuted after
        # quantization is finished
        actorder = False
        if hasattr(self.layer, "quantization_scheme"):
            quant_scheme = self.layer.quantization_scheme
            quant_weights = quant_scheme.weights
            if quant_weights is not None:
                actorder = quant_weights.actorder
                if actorder:
                    # use hessian to create a permutation of weights
                    perm = torch.argsort(torch.diag(H), descending=True)

                    # permute weight and hessian
                    W = W[:, perm]
                    H = H[perm][:, perm]

            # fetch latest correct scale and ZP relevant for any changes
            from compressed_tensors.quantization import update_layer_weight_quant_params

            # TODO: experiment with updating before each block
            update_layer_weight_quant_params(self.layer, weight=W, reset_obs=True)
            scale = self.layer.weight_scale.data
            zero_point = self.layer.weight_zero_point.data

        group_size = (
            quant_scheme.weights.group_size
            if quant_scheme.weights.group_size is not None
            else W.shape[1]
        )

        # mask sparsity if applicable
        sparsity = tensor_sparsity(W)
        preserve_zeros = sparsity >= SPARSITY_THRESHOLD
        W_nz_mask = (
            (~torch.isclose(W, torch.zeros(1, device=W.device).float())).float()
            if preserve_zeros
            else None
        )

        # invalidate dead hessian values
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        # compute hessian inverse
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # See section 3.4 of https://arxiv.org/abs/2203.07259
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if preserve_zeros:
                W1_nz_mask = W_nz_mask[:, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()

                if hasattr(self.layer, "weight_fake_quant"):
                    scale = self.layer.weight_fake_quant.scale
                    zero_point = self.layer.weight_fake_quant.zero_point
                    dtype = self.layer.weight_fake_quant.dtype
                    qscheme = self.layer.weight_fake_quant.qscheme
                    if qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                        q = torch.quantize_per_tensor(q, scale, zero_point, dtype)
                    else:
                        q = torch.quantize_per_channel(q, scale, zero_point, 0, dtype)
                    q = torch.dequantize(q)
                elif hasattr(self.layer, "quantization_scheme"):
                    quant_scheme = self.layer.quantization_scheme

                    if quant_scheme.weights is not None:
                        from compressed_tensors.quantization import QuantizationStrategy
                        from compressed_tensors.quantization.lifecycle.forward import (
                            fake_quantize,
                        )

                        strategy = quant_scheme.weights.strategy

                        if strategy == QuantizationStrategy.TENSOR:
                            q = fake_quantize(
                                q,
                                scale,
                                zero_point,
                                self.layer.quantization_scheme.weights,
                            )
                        elif strategy == QuantizationStrategy.CHANNEL:
                            # TODO: for channelwise why isn't this just a 1d tensor?
                            q = fake_quantize(
                                q,
                                scale[:, 0],
                                zero_point[:, 0],
                                quant_scheme.weights,
                            )
                        else:  # strategy == QuantizationStrategy.GROUP
                            # get the group index for the current column
                            column_idx = i1 + i
                            input_dim_group = column_idx // group_size

                            # Since we're only applying quantization to a slice, this
                            # ends up being a channelwise application
                            altered_qargs = copy(quant_scheme.weights)
                            altered_qargs.strategy = QuantizationStrategy.CHANNEL

                            q = fake_quantize(
                                q,
                                scale[:, input_dim_group],
                                zero_point[:, input_dim_group],
                                altered_qargs,
                            )

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                if preserve_zeros:
                    W1[:, i:] -= w1_err * W1_nz_mask[:, i:]
                else:
                    W1[:, i:] -= w1_err
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            w_err = Err1.matmul(Hinv[i1:i2, i2:])
            if preserve_zeros:
                W[:, i2:] -= w_err * W_nz_mask[:, i2:]
            else:
                W[:, i2:] -= w_err
        logger.info("time %.2f" % (time.time() - tick))
        logger.info("error %.2f" % torch.sum(Losses).item())

        if actorder:
            # restore original permutation
            invperm = torch.argsort(perm)
            W = W[:, invperm]

            # g_idx describes the group index of the permuted weight
            g_idx = torch.tensor(
                [i // group_size for i in range(self.columns)],
                dtype=torch.int,
            ).to(device=invperm.device)

            # invert to get the group index of the unpermuted weight
            self.layer.weight_g_idx.data = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.reshape(final_shape).to(final_dtype)

        # This is a bit hacky, but FSDP updates only work if we change
        # the weight in place, clone() or direct assignment won't work
        self.layer.weight -= self.layer.weight
        self.layer.weight += W

        if is_module_offloaded(self.layer):
            device = get_offloaded_device(self.layer)
            update_prefix_dict(self.layer, "weight", self.layer.weight.to(device))
            self.layer._hf_hook.post_forward(self.layer, None)

        del W
        del Losses
        del diag

    def free(self):
        """
        Free the Hessian memory after the layer is complete
        """
        delattr(self, "H")
        super().free()
