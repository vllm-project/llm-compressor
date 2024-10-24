import contextlib
from abc import abstractmethod
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Tuple

import torch
from loguru import logger
from pydantic import BaseModel
from torch.utils.hooks import RemovableHandle

from llmcompressor.utils.helpers import getattr_chain
from llmcompressor.utils.metric_logging import CompressionLogger
from llmcompressor.utils.pytorch.module import get_layers, get_no_split_params

__all__ = ["HooksMixin", "LayerCompressorMixin"]


class HooksMixin(BaseModel):
    """ "
    Class to manage the registration, disabling, and removal of hooks. Registering
    and removing hooks should be handled by modifier classes which inherit from this
    mixin, while disabling hooks should disable all hooks across modifiers.

    Modifiers which implement hooks should use the @HooksMixin.hook decorator
    Modifiers must pass registered hooks handles to self.register_hook() and must
    remove hooks when finished using self.remove_hooks()
    """

    _HOOKS_DISABLED: ClassVar[bool] = False
    _hooks: List[RemovableHandle] = []

    @classmethod
    def hook(cls, func: Callable[[Any], Any]):
        def wrapped(*args, **kwargs):
            if cls._HOOKS_DISABLED:
                return

            func(*args, **kwargs)

        return wrapped

    @classmethod
    @contextlib.contextmanager
    def disable_hooks(cls):
        """
        Disable all hooks across all modifiers
        TODO: select which modifier hooks are disabled/ kept enabled
        """
        try:
            cls._HOOKS_DISABLED = True
            yield
        finally:
            cls._HOOKS_DISABLED = False

    def register_hook(self, handle: RemovableHandle):
        """
        Usage: self.register_hook(module.register_forward_hook(...))

        :param handle: handle of added hook
        """
        self._hooks.append(handle)

    def remove_hooks(self):
        """
        Remove all hooks belonging to a modifier
        """
        for hook in self._hooks:
            hook.remove()


class LayerCompressorMixin(HooksMixin):
    """
    Apply a given compression function to a model during the model's calibration
    forward pass

    Lifecycle:
        - QuantizationModifier.initialize(model)
        - SequentialLayerCompressor(compress_fn)
        - register_hooks(model)
        - model.forward()
            - compress_fn(name, target_module, args)
        - remove_hooks()

    :ivar true_sequential: Used to control the granularity of compression updates
        through the forward pass. Set to True to use the weight-compressed outputs
        of each module, set to False to use the weight-compressed outputs of each
        layer (transformer block), defaults to False
    :ivar sequential_targets: list of layer names to compress during GPTQ, or
        '__ALL__' to compress every layer in the model
    :ivar compresss_module: Function to be called on target modules
    """

    true_sequential: bool
    sequential_targets: bool
    # compress_module: Callable[[str, torch.nn.Module, Tuple], float]

    _layer_index = 0
    _num_layers = 0

    @abstractmethod
    def compress_module(
        self,
        name: str,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
    ) -> float:
        raise NotImplementedError()

    def register_hooks(self, model: torch.nn.Module):
        # find layers (used for printing even if true_sequential=True)
        # if no targets are provided, default to the modules that shouldn't be
        # split by FSDP. For Transformers models this is equivalent to the
        # decoder layers (ie LlamaDecoderLayer)
        sequential_targets = self.sequential_targets
        if sequential_targets is None:
            sequential_targets = get_no_split_params(model)
        layers = get_layers(sequential_targets, model)
        self._num_layers = len(layers)

        for name, module in model.named_modules():
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                pre_hook = partial(self.target_pre_forward, name)
                post_hook = partial(self.target_post_forward, name)
                self.register_hook(module.register_forward_pre_hook(pre_hook))
                self.register_hook(module.register_forward_hook(post_hook))

            if name in layers.keys():
                pre_hook = partial(self.layer_pre_forward, name)
                post_hook = partial(self.layer_post_forward, name)
                self.register_hook(module.register_forward_pre_hook(pre_hook))
                self.register_hook(
                    module.register_forward_hook(post_hook, with_kwargs=True)
                )

    @HooksMixin.hook
    def target_pre_forward(
        self, name: str, module: torch.nn.Module, args: Tuple[Any, ...]
    ):
        if self.true_sequential:
            # compress first so output is from compressed weights
            with CompressionLogger(module) as comp_logger:
                loss = self.compress_module(name, module, args)
                comp_logger.set_loss(loss)

    @HooksMixin.hook
    def target_post_forward(
        self,
        name: str,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        _output: Tuple[Any, ...],
    ):
        if not self.true_sequential:
            # compress after so output is from uncompressed weights
            with CompressionLogger(module) as comp_logger:
                loss = self.compress_module(name, module, args)
                comp_logger.set_loss(loss)

    @HooksMixin.hook
    def layer_pre_forward(self, _name: str, _module: torch.nn.Module, _args: Any):
        logger.info(
            f"\n===== Compressing layer {self._layer_index}/{self._num_layers} ====="
        )

    @HooksMixin.hook
    def layer_post_forward(
        self,
        _name: str,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        output: Tuple[Any, ...],
    ):
        if not self.true_sequential:
            # rerun with (now) compressed weights
            with HooksMixin.disable_hooks():
                compressed_output = module(*args, **kwargs)

            error = torch.nn.functional.l1_loss(output[0], compressed_output[0])
            logger.info(f"Mean output error from quantization: {error:.3f}")

        self._layer_index += 1
        return output
