import contextlib
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.quantization import (
    QuantizationScheme,
    apply_quantization_config,
)
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    getattr_chain,
    update_offload_parameter,
)
from loguru import logger
from pydantic import Field, PrivateAttr, field_validator

from llmcompressor.core import Event, EventType, State
from llmcompressor.core.utils import resolve_modifier_quantization_config
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import freeze_module_quantization
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.utils.metric_logging import CompressionLogger

__all__ = ["GPTQModifier"]


class GPTQModifier(Modifier, HooksMixin):
    """
    Implements the GPTQ algorithm from https://arxiv.org/abs/2210.17323. This modifier
    uses activations to calibrate a hessian matrix, which is then used to determine
    optimal quantizion values and orderings for the model weights.

    | Sample yaml:
    | test_stage:
    |    obcq_modifiers:
    |      GPTQModifier:
    |          block_size: 128
    |          dampening_frac: 0.001
    |          offload_hessians: False
    |          config_groups:
    |            group_0:
    |                targets:
    |                  - "Linear"
    |                input_activations: null
    |                output_activations: null
    |                weights:
    |                    num_bits: 8
    |                    type: "int"
    |                    symmetric: true
    |                    strategy: "tensor"
    |                    group_size: 128
    |                    actorder: False

    Lifecycle:
        - on_initialize
            - _build_quant_modifier
            - register_hook(module, compress_module, "forward")
            - run_sequential / run_layer_sequential / run_basic
                - make_empty_hessian
                - accumulate_hessian
        - on_sequential_batch_end
            - quantize_weight
        - on_finalize
            - remove_hooks()
            - model.apply(freeze_module_quantization)

    :param sequential_targets: list of layer names to compress during GPTQ, or
        '__ALL__' to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param quantize: Set to True to quantize using an existing quantization modifier,
        or pass in the configuration for a quantization modifier if one does not
        already exist in the recipe
    :param offload_hessians: Set to True for decreased memory usage but increased
        runtime.
    :param config_groups: [Used, if a quantization modifier is not specified],
        dictionary specifying quantization schemes to apply to target
        modules. Modules not matching a scheme target will NOT be quantized.
    :param scheme: [Used, if a quantization modifier is not specified], the quantization
        scheme to apply to the model, this is a dictionary that supports all keys from
        QuantizationScheme except targets, which will be set to the targets parameter
        set at the modifier level. Can also be set to a dictionary of the format
        `preset_scheme_name: targets` for example: `W8A8: ['Linear']` for weight 8 bit
        or a string of a preset scheme if targets is provided
        and activation 8 bit quantization on the Linear layers.
    :param targets: list of layer names to quantize if a scheme is provided. Defaults
        to Linear layers
    :param ignore: [Used, if a quantization modifier is not specified]
        optional list of module class names or submodule names to not
        quantize even if they match a target in config_groups. Defaults to empty list.
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    """

    # gptq modifier arguments
    sequential_update: bool = True  # DEPRECIATED
    sequential_targets: Union[str, List[str], None] = None
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    quantize: Union[bool, Dict] = True
    offload_hessians: bool = False

    # arguments used for attached quant modifier
    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    targets: Union[str, List[str], None] = None
    ignore: List[str] = Field(default_factory=list)
    num_calibration_steps: Optional[int] = None

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)

    @field_validator("targets", mode="before")
    def validate_targets(cls, value: Union[str, List[str]]) -> List[str]:
        if isinstance(value, str):
            return [value]

        return value

    @field_validator("sequential_update", mode="before")
    def validate_sequential_update(cls, value: bool) -> bool:
        if not value:
            warnings.warn(
                "`sequential_update=False` is no longer supported, setting "
                "sequential_update=True",
                DeprecationWarning,
            )

        return True

    def on_initialize(self, state: State) -> bool:
        """
        Initialize and run the GPTQ algorithm on the current state

        :param state: state storing input model and calibration data
        """
        # build quantization modifier
        config = resolve_modifier_quantization_config(self)
        apply_quantization_config(state.model, config)

        # prepare module names
        self._module_names = {m: name for name, m in state.model.named_modules()}

        # register hooks
        for module in state.model.modules():
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                # HACK: previously, embeddings were not quantized because they were not
                # accessible by the layer compressor. For now, we manually ignore it,
                # but in the FUTURE this should be ignored by the user
                if not isinstance(module, torch.nn.Embedding):
                    self.register_hook(module, self.calibrate_module, "forward")

        return True

    def on_event(self, state: State, event: Event):
        if event.type_ in (
            EventType.SEQUENTIAL_EPOCH_END,
            EventType.CALIBRATION_EPOCH_END,
        ):
            self.compress_modules()

    def on_finalize(self, state: State) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: state storing input model and calibration data
        """
        if len(self._num_samples) > 0:
            raise ValueError(f"Failed to compress {len(self._num_samples)} modules")

        self.remove_hooks()
        self._hessians = dict()
        self._num_samples = dict()
        state.model.apply(freeze_module_quantization)

        return True

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        """
        Calibration hook used to accumulate the hessian of the input to the module

        :param module: module being calibrated
        :param args: inputs to the module, the first element of which is the
            cannonical input
        :param _output: uncompressed module output, unused
        """
        # Assume that first argument is the input
        inp = args[0]

        # Initialize hessian if not present
        if module not in self._num_samples:
            init_device = (
                "cpu" if self.offload_hessians else get_execution_device(module)
            )
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = 0

        # Accumulate hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

    def compress_modules(self):
        """
        Quantize modules which have been calibrated
        """
        for module in list(self._num_samples.keys()):
            name = self._module_names[module]
            num_samples = self._num_samples[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            logger.info(f"Quantizing {name} using {num_samples} samples")
            with (
                torch.no_grad(),
                align_module_device(module),
                self._maybe_onload_hessian(module),
                CompressionLogger(module) as comp_logger,
            ):
                loss, quantized_weight, scale, zero_point, g_idx = quantize_weight(
                    module=module,
                    quant_args=quant_args,
                    hessians_dict=self._hessians,
                    blocksize=self.block_size,
                    percdamp=self.dampening_frac,
                )
                comp_logger.set_loss(loss)

            update_offload_parameter(module, "weight", quantized_weight)
            update_offload_parameter(module, "weight_scale", scale)
            update_offload_parameter(module, "weight_zero_point", zero_point)
            if g_idx is not None:
                update_offload_parameter(module, "weight_g_idx", g_idx)

            # self._hessians[module] already deleted by quantize_weight
            del self._num_samples[module]

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:  # may have been deleted in context
                self._hessians[module] = self._hessians[module].to(device="cpu")
