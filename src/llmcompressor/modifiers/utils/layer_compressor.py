import logging
import operator
from typing import Dict

import torch
from torch.nn import Module

from llmcompressor.modifiers.utils.compression_wrapper import ModuleCompressionWrapper
from llmcompressor.utils.fsdp.context import (
    fix_fsdp_module_name,
    summon_full_params_context,
)
from llmcompressor.utils.pytorch import set_layer
from llmcompressor.utils.pytorch.module import get_prunable_layers

__all__ = ["LayerCompressor"]

_LOGGER = logging.getLogger(__name__)


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
            - module_compressor_class.fasterprune()
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
        self.modules = {}

    def compressible_modules(self) -> Dict:
        """
        Get the list of modules in the layer that can be compressed

        :return: dictionary of compressible modules
        """
        compressible_layers = get_prunable_layers(self.layer)
        return compressible_layers

    def pre_compress(self):
        """
        Sets up the SparseGPT objects for each compressible module, computes the Hessian
        for each using the calibration data.
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
            module_wrapper.free()
        self.modules = None

    def compress(self):
        """
        Apply compression to each wrapped submodule in the layer
        """

        @torch.no_grad()
        def prune(module):
            if isinstance(module, self.module_compressor_class):
                full_name = self._get_full_submodule_name(module.name)
                _LOGGER.info(f"Compressing {full_name}...")
                module.fasterprune(**self.args)

        self.layer.apply(prune)

    def _get_full_submodule_name(self, name):
        full_name = ".".join(x for x in [self.name, name] if len(x) > 0)
        full_name = fix_fsdp_module_name(full_name)
        return full_name
