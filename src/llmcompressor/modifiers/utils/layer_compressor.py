import operator
from typing import Dict, Tuple

import torch
from compressed_tensors import get_execution_device
from loguru import logger
from torch.nn import Module
from tqdm import tqdm

from llmcompressor.modifiers.utils.compression_wrapper import ModuleCompressionWrapper
from llmcompressor.modifiers.utils.pytorch_helpers import EarlyStopException
from llmcompressor.pytorch.utils import tensors_to_device
from llmcompressor.utils.fsdp.context import (
    fix_fsdp_module_name,
    summon_full_params_context,
)
from llmcompressor.utils.pytorch import set_layer
from llmcompressor.utils.pytorch.module import get_prunable_layers

__all__ = ["LayerCompressor"]


class LayerCompressor:
    """
    Runs weight sparisification on a single layer using calibration data inputs. The
    layer may contain submodules. The specific sparsification algorithm is determined
    by module_compressor_class.

    Lifecycle:
        - pre_compress()
            - compressible_modules()
            - module_compressor_class.register_forward_hook()
        - compress()
            - module_compressor_class.compress()
        - post_compress()
        - revert_layer_wrappers()

    :param module_compressor_class: wrapper class to use for root modules
    :param model: model containing the layer we are running compression on
    :param layer: layer to run compression on
    :param layer_index: index of layer in the model
    :param args: additional keyword arguments
    """

    def __init__(
        self,
        module_compressor_class: ModuleCompressionWrapper,
        model: Module,
        layer: Module,
        layer_index: int,
        name: str,
        args: Dict,
    ):
        self.module_compressor_class = module_compressor_class
        self.model = model
        self.layer = layer
        self.layer_index = layer_index
        self.name = name
        self.args = args
        self.handles = None
        self.early_stop_handle = None
        self.modules = {}

    def compressible_modules(self) -> Dict:
        """
        Get the list of modules in the layer that can be compressed

        :return: dictionary of compressible modules
        """
        compressible_layers = get_prunable_layers(self.layer)
        return compressible_layers

    def set_early_stop(self):
        """
        Adds an early stopping exception to the input of the layer. This will cause the
        model to immediately exit the forward pass when reaching this layer.
        """

        def trigger_early_stop_fn(self, args, kwargs):
            raise EarlyStopException(args, kwargs)

        self.early_stop_handle = self.layer.register_forward_pre_hook(
            trigger_early_stop_fn, with_kwargs=True
        )

    def clear_early_stop(self):
        """
        Clears the early stopping handle
        """
        if self.early_stop_handle is not None:
            self.early_stop_handle.remove()
            self.early_stop_handle = None

    def pre_compress(self):
        """
        Sets up the CompressionWrapper objects for each compressible module, adding a
        hook for computing the Hessians as calibration data is passed through.
        """
        subset = self.compressible_modules()

        for name in subset:
            layer = subset[name]
            full_name = self._get_full_submodule_name(name)
            with summon_full_params_context(self.layer):
                wrapper = self.module_compressor_class(full_name, layer)
            if len(name) == 0:  # special case if layer has no children (i.e. lm_head)
                with summon_full_params_context(self.model):
                    set_layer(full_name, wrapper, self.model)
            else:
                set_layer(name, wrapper, self.layer)
            self.modules[name] = wrapper

        self.layer = operator.attrgetter(self.name)(self.model)

        def add_batch(name):
            def tmp(_, inp, out):
                self.modules[name].add_batch(inp[0].data, out.data)

            return tmp

        self.handles = []
        for name in self.modules:
            self.handles.append(subset[name].register_forward_hook(add_batch(name)))

    def calibrate_layer(self, intermediates: Tuple[Tuple, Dict]) -> Tuple[Tuple, Dict]:
        """
        Runs all calibration samples through the stored layer

        :param intermediates: inputs to run through the layer
        :return: outputs of the layer
        """
        for idx in tqdm(range(len(intermediates))):
            args, kwargs = intermediates[idx]
            device = get_execution_device(self.layer)
            output = self.layer(*tensors_to_device(args, device), **kwargs)
            intermediates[idx] = (tensors_to_device(output, "cpu"), kwargs)
            torch.cuda.empty_cache()

        return intermediates

    def post_compress(self):
        """
        remove the add_batch forward hooks after compression is complete
        """
        for handle in self.handles:
            handle.remove()

    def revert_layer_wrappers(self):
        """
        Reverts wrapped root modules back to their original structure
        """
        for name, module_wrapper in self.modules.items():
            full_name = self._get_full_submodule_name(name)
            if len(name) == 0:  # special case if layer has no children (i.e. lm_head)
                with summon_full_params_context(self.model):
                    set_layer(full_name, module_wrapper.layer, self.model)
            else:
                set_layer(name, module_wrapper.layer, self.layer)
            torch.cuda.empty_cache()
        self.modules = None

    def compress(self):
        """
        Apply compression to each wrapped submodule in the layer
        """

        @torch.no_grad()
        def compress_module(module):
            if isinstance(module, self.module_compressor_class):
                full_name = self._get_full_submodule_name(module.name)
                logger.info(f"Compressing {full_name}...")
                module.compress(**self.args)
                module.free()
                print("done")

        self.layer.apply(compress_module)
        torch.cuda.empty_cache()

    def _get_full_submodule_name(self, name):
        full_name = ".".join(x for x in [self.name, name] if len(x) > 0)
        full_name = fix_fsdp_module_name(full_name)
        return full_name
