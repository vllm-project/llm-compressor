import contextlib
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    getattr_chain,
    update_offload_parameter,
)
from loguru import logger
from pydantic import Field, PrivateAttr, field_validator

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier, ModifierFactory
from llmcompressor.modifiers.quantization.calibration import freeze_module_quantization
from llmcompressor.modifiers.quantization.gptq.utils.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.modifiers.quantization.quantization.base import QuantizationModifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.layer_sequential import (
    run_pipeline as run_layer_sequential,
)
from llmcompressor.pipelines.sequential import run_pipeline as run_sequential
from llmcompressor.utils.metric_logging import CompressionLogger
from llmcompressor.utils.pytorch.module import get_no_split_params, qat_active

__all__ = ["GPTQModifier"]


class GPTQModifier(Modifier, HooksMixin):
    """
    Modifier for applying the one-shot OBCQ algorithm to a model

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
        - on_initialize_structure
            - _build_quant_modifier
        - on_initialize
            - register_hook(module, compress_module, "forward")
            - run_sequential / run_layer_sequential
                - make_empty_hessian
                - accumulate_hessian
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
    :param disable_quantization_observer_epoch: [Used, if a quantization modifier is
        not specified] Epoch to disable updates to the module
        quantization observers. At this point, quantized weights and zero points will
        not be updated. Leave None to not disable observers during QAT. Default is None
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
    disable_quantization_observer_epoch: Optional[float] = None

    # private variables
    _quantization_modifier: Optional[QuantizationModifier] = PrivateAttr(default=None)
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)

    @field_validator("sequential_update", mode="before")
    def validate_sequential_update(cls, value: bool) -> bool:
        if not value:
            warnings.warn(
                "`sequential_update=False` is no longer supported, setting "
                "sequential_update=True",
                DeprecationWarning,
            )

        return True

    def on_initialize_structure(self, state: State, **kwargs):
        """
        Check the model's quantization state matches that expected by this modifier,
        adding a default quantization scheme if needed

        TODO: Depreciate and fold into `on_initialize`

        :param state: session state storing input model and calibration data
        """
        quantization_already_active = qat_active(state.model)
        if isinstance(self.quantize, bool):
            if not self.quantize and quantization_already_active:
                logger.warning(
                    "GPTQ quantization is set to False, but a "
                    "quantization modifier is already active on the model "
                    "resetting quantize to True"
                )
                self.quantize = True
            elif self.quantize and not quantization_already_active:
                logger.warning(
                    "GPTQ quantization is set to True without an "
                    "active quantization modifier."
                )
                self._build_quant_modifier()
            return  # use existing quantization modifier if there is one
        else:
            if not isinstance(self.quantize, Dict):
                raise ValueError(
                    "GPTQModifier.quantize accepts only a single "
                    "quantization modifier or a boolean. Found "
                    f"type {type(self.quantize)}"
                )
            if len(self.quantize) != 1:
                raise ValueError(
                    "GPTQModifier.quantize accepts only a single "
                    "quantization modifier or a boolean. Found "
                    f"{len(self.quantize)} modifiers"
                )
            if quantization_already_active:
                logger.warning(
                    "Attempting to initialize quantization for GPTQ "
                    "but a quantization modifier has already been applied. "
                    "The quantization configuration defined under the "
                    "GPTQ modifier will be ignored."
                )
                self.quantize = True
                return
            self._build_quant_modifier_from_dict(self.quantize)
            self.quantize = True

        if self._quantization_modifier:
            self._quantization_modifier.on_initialize_structure(state, **kwargs)

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the GPTQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        # initialize quantization modifier
        if not self.initialized_structure_:
            self.on_initialize_structure(state, **kwargs)
        if self._quantization_modifier:
            self._quantization_modifier.initialize(state, **kwargs)
        if not self.quantize:
            raise ValueError("To use the GPTQModifier, quantization must be enabled.")

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

        # infer sequential targets
        if self.sequential_targets is None:
            self.sequential_targets = get_no_split_params(state.model)
        if isinstance(self.sequential_targets, str):
            self.sequential_targets = [self.sequential_targets]

        # infer pipeline
        model_name = state.model.__class__.__name__
        input_names = state.data.calib.dataset.column_names
        unfixable_errors = (
            torch.OutOfMemoryError,
            torch._C._LinAlgError,
            KeyboardInterrupt,
        )
        try:
            run_sequential(
                state.model,
                state.data.calib,
                self.sequential_targets,
                self.ignore,
                self,
            )
            return True

        except Exception as exception:
            if isinstance(exception, torch.fx.proxy.TraceError):
                warnings.warn(f"Failed to trace {model_name} with inputs {input_names}")
            if isinstance(exception, unfixable_errors):
                raise exception

            warnings.warn("Falling back to layer_sequential pipeline")
            run_layer_sequential(
                state.model,
                state.data.calib,
                self.sequential_targets,
                self,
            )
            return True

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        if self._quantization_modifier:
            self._quantization_modifier.finalize(state, **kwargs)

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
        Quantize a module's weight according to the GPTQ algorithm

        :param name: name of module being quantized
        :param module: module being quantized
        :param args: input arguments for module forward pass

        :return: total loss from applying weight quantization to this module
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

    def quantize_modules(self, model: torch.nn.Module):
        for module in model.modules():
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

        assert len(self._hessians) <= 0
        assert len(self._num_samples) <= 0

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:  # may have been deleted in context
                self._hessians[module] = self._hessians[module].to(device="cpu")

    def _build_quant_modifier(self):
        """
        Build a quantization modifier based on the specified config_groups,
        ignore list, and num_calibration_steps.

        :postcondition: self._quantization_modifier is set to the built
            quantization modifier
        """

        quantization_args_names = [
            "config_groups",
            "targets",
            "scheme",
            "num_calibration_steps",
            "ignore",
            "disable_quantization_observer_epoch",
        ]

        quant_args = {
            key: getattr(self, key)
            for key in quantization_args_names
            if getattr(self, key, False)
        }

        logger.info(f"Building quantization modifier with args: {quant_args}")
        vllm_quant_config = {"QuantizationModifier": quant_args}
        self._build_quant_modifier_from_dict(vllm_quant_config)

    def _build_quant_modifier_from_dict(self, quant_config):
        modifier_type = list(quant_config.keys())[0]
        modifier_args = quant_config[modifier_type]
        self._quantization_modifier = ModifierFactory.create(
            modifier_type,
            allow_registered=True,
            allow_experimental=True,
            **modifier_args,
        )
