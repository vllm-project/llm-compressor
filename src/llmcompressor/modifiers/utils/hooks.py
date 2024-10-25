import contextlib
from abc import abstractmethod
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Set, Tuple, Union

import torch
from loguru import logger
from pydantic import BaseModel
from torch.utils.hooks import RemovableHandle
from collections import defaultdict

from llmcompressor.modifiers.quantization.gptq.utils.gptq_quantize import add_batch
from llmcompressor.modifiers.utils.pytorch_helpers import EarlyStopException
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

    Lifecycle:
        - Modifier.register_hooks(model)
        - model.forward()
        - Modifier.remove_hooks()
    """

    _HOOKS_DISABLED: ClassVar[bool] = False
    _hooks: List[RemovableHandle] = []

    @classmethod
    def hook(cls, func: Callable[[Any], Any]):
        def wrapped(*args, **kwargs):
            if cls._HOOKS_DISABLED:
                return

            return func(*args, **kwargs)

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
        - Modifier.register_hooks(model)
        - model.forward()
            - compress_fn(name, target_module, args)
        - Modifier.remove_hooks()

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
    _pre_active: Set[torch.nn.Module] = set()
    _module_inputs: Dict[torch.nn.Module, List[Tuple[Tuple[Any, ...], Dict[str, Any]]]] = defaultdict(lambda: [])
    _module_outputs: Dict[torch.nn.Module, Union[List[Tuple[Any, ...]], torch.Tensor]] = defaultdict(lambda: [])

    _layer_inputs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    _layer_outputs: List[Tuple[Any, ...]] = []

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
                self.register_hook(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
                self.register_hook(module.register_forward_hook(post_hook))

            if name in layers.keys():
                pre_hook = partial(self.layer_pre_forward, name)
                post_hook = partial(self.layer_post_forward, name)
                self.register_hook(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
                self.register_hook(
                    module.register_forward_hook(post_hook, with_kwargs=True)
                )
        

    @HooksMixin.hook
    def target_pre_forward(
        self, name: str, module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ):
        input = args[0]

        # compute hessian
        if not hasattr(module, "gptq_hessian"):
            num_columns = module.weight.shape[1]
            module.gptq_hessian = torch.zeros((num_columns, num_columns), dtype=torch.float32, device=input.device)
            module.gptq_hessian_samples = 0

        print(f"{name} adding {input.size(0)} samples")
        module.gptq_hessian, module.gptq_hessian_samples = add_batch(
            module.gptq_hessian,
            module.gptq_hessian_samples,
            module,
            input
        )


    @HooksMixin.hook
    def target_post_forward(
        self,
        name: str,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        output: Tuple[Any, ...],
    ):
        print(f"post {name}")

        if module.gptq_hessian_samples >= 20:
            # compress
            print(f"compressing {name}")
            if True: #self.true_sequential:
                with CompressionLogger(module) as comp_logger:
                    loss = self.compress_module(name, module, args)
                    comp_logger.set_loss(loss)

        """
        breakpoint()
        ret = torch.concat(self._module_outputs)
        del self._module_inputs[module] 
        del self._module_outputs[module]
        return ret

        # accumulate
        self._module_outputs.append(output)

        if len(self._module_outputs) == 2:
            with CompressionLogger(module) as comp_logger:
                loss = self.compress_module(name, module, args)
                comp_logger.set_loss(loss)

        ret = self._module_outputs
        self._module_outputs = []

        return ret

        if self.true_sequential:
            # compress first so output is from compressed weights
            with CompressionLogger(module) as comp_logger:
                loss = self.compress_module(name, module, args)
                comp_logger.set_loss(loss)

        if not self.true_sequential:
            # compress after so output is from uncompressed weights
            with CompressionLogger(module) as comp_logger:
                loss = self.compress_module(name, module, args)
                comp_logger.set_loss(loss)
        """

    @HooksMixin.hook
    def layer_pre_forward(self, name: str, layer: torch.nn.Module, args: Any, kwargs):
        logger.info(
            f"\n===== Compressing layer {self._layer_index}/{self._num_layers} ====="
        )

        input = args[0]

        if not self.true_sequential:
            self._module_inputs[layer] += [
                input[batch_index: batch_index + 1]
                for batch_index in range(input.shape[0])
            ]


        if len(self._module_outputs[layer]) >= 20 - 1:
            # last sample can be passed normally
            print("last layer forward")
            return (input[-1:], *args[1:]), kwargs
        
        else:
            forward_call = (layer._slow_forward if torch._C._get_tracing_state() else layer.forward)
            for batch_index in range(input.size(0)):
                print("layer forward")
                output = forward_call(input[batch_index: batch_index + 1], *args[1:], **kwargs)
                self._module_outputs[layer].append(output)

            raise EarlyStopException(torch.tensor([]), None)


    @HooksMixin.hook
    def layer_post_forward(
        self,
        name: str,
        layer: torch.nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        output: Tuple[Any, ...],
    ):
        print(f"post {name}")
        breakpoint()

        # capture last sample
        self._module_outputs[layer].append(output)

        # batch outputs
        outputs = self._module_outputs[layer]
        batched_outputs = tuple(
            torch.concat(tuple(
                outputs[sample_index][output_index]
                for sample_index in range(len(outputs))
            ))
            for output_index in range(len(outputs[0]))
        )
        del self._module_outputs[layer]

        if not self.true_sequential:
            pass  # run again

            del self._module_inputs[layer] 

        return batched_outputs
    
        if not self.true_sequential:
            # rerun with (now) compressed weights
            with HooksMixin.disable_hooks():
                compressed_output = layer(*args, **kwargs)

            error = torch.nn.functional.l1_loss(output[0], compressed_output[0])
            logger.info(f"Mean output error from quantization: {error:.3f}")

        self._layer_index += 1
        return output
