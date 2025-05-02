import contextlib
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.quantization import disable_quantization
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    getattr_chain,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr, field_validator

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.pipelines.basic import run_pipeline as run_basic
from llmcompressor.pipelines.layer_sequential import (
    run_pipeline as run_layer_sequential,
)
from llmcompressor.pipelines.sequential import run_pipeline as run_sequential
from llmcompressor.utils.metric_logging import CompressionLogger
from llmcompressor.utils.pytorch.module import get_no_split_params

__all__ = ["GPTQModifier"]


class GPTQModifier(Modifier, QuantizationMixin):
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
    :param offload_hessians: Set to True for decreased memory usage but increased
        runtime.

    :param config_groups: dictionary specifying quantization schemes to apply to target
        modules. Modules not matching a scheme target will NOT be quantized.
    :param targets: list of layer names to quantize if a scheme is provided. Defaults
        to Linear layers
    :param ignore: optional list of module class names or submodule names to not
        quantize even if they match a target in config_groups. Defaults to empty list.
    :param scheme: a single quantization scheme to apply to the model. This is a
        dictionary that supports all keys from QuantizationScheme except targets, which
        will be set to the targets parameter set at the modifier level. Can also be set
        to a dictionary of the format `preset_scheme_name: targets` for example:
        `W8A8: ['Linear']` for weight and activation 8-bit.
    :param kv_cache_scheme: optional QuantizationArgs, that specify the
        quantization of the kv cache. If None, kv cache is not quantized.
        When applying kv cache quantization to transformer AutoModelForCausalLM,
        the kv_cache_scheme gets converted into a QuantizationScheme that:
            - targets the `q_proj` and `k_proj` modules of the model. The outputs
              of those modules are the keys and values that might be cached
            - quantizes the outputs of the aformentioned layers, so that
              keys and values are compressed before storing them in the cache
        There is an explicit assumption that the model contains modules with
        `k_proj` and `v_proj` in their names. If this is not the case
        and kv_cache_scheme != None, the quantization of kv cache will fail
    """

    # gptq modifier arguments
    sequential_update: bool = True  # DEPRECIATED
    sequential_targets: Union[str, List[str], None] = None
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    quantize: Union[bool, Dict] = True
    offload_hessians: bool = False

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)

    @field_validator("sequential_update", mode="before")
    def validate_sequential_update(cls, value: bool) -> bool:
        if not value:
            warnings.warn(
                "`sequential_update=False` is no longer supported, setting "
                "sequential_update=True",
                DeprecationWarning,
            )

        return True

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the GPTQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # assume quantization has been initialized by this modifier or one before it
        QuantizationMixin.start_calibration(self, state.model)
        # Unlike qmod, do not quantize as we calibrate
        # This choice does not seem to have a meaningful impact on accuracy
        state.model.apply(disable_quantization)

        # prepare module names
        self._module_names = {m: name for name, m in state.model.named_modules()}

        # register hooks
        added_hook = False
        for module in state.model.modules():
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                # HACK: previously, embeddings were not quantized because they were not
                # accessible by the layer compressor. For now, we manually ignore it,
                # but in the FUTURE this should be ignored by the user
                if not isinstance(module, torch.nn.Embedding):
                    self.register_hook(module, self.calibrate_module, "forward")
                    added_hook = True

        if not added_hook:
            raise ValueError(
                "GPTQModifier requires a quantization config be specified by this "
                "modifier or a modifier preceding it"
            )

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
                warnings.warn(
                    f"Failed to trace {model_name} with inputs {input_names}. For more "
                    "information on tracing with the sequential pipeline, see "
                    "https://github.com/vllm-project/llm-compressor/blob/main/"
                    "src/llmcompressor/transformers/tracing/GUIDE.md"
                )
            if isinstance(exception, unfixable_errors):
                raise exception

            warnings.warn("Falling back to layer_sequential pipeline")
            try:
                run_layer_sequential(
                    state.model,
                    state.data.calib,
                    self.sequential_targets,
                    self,
                )
                return True

            except Exception as exception:
                if isinstance(exception, TypeError):
                    warnings.warn(f"{model_name} fails layer-wise assumptions")
                if isinstance(exception, unfixable_errors):
                    raise exception

                warnings.warn(
                    "Falling back to basic pipeline, which requires extra memory and "
                    "may result in decreased accuracy. Consider using "
                    "`offload_hessians=True`"
                )
                run_basic(state.model, state.data.calib, self)
                return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        self._hessians = dict()
        self._num_samples = dict()

        QuantizationMixin.end_calibration(self, state.model)
        self.remove_hooks()  # remove gptq hooks

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

    def on_sequential_batch_end(self):
        """
        Quantize modules.
        TODO: implement with event callback
        """
        for module in list(self._num_samples.keys()):
            name = self._module_names[module]
            num_samples = self._num_samples[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            logger.info(f"Quantizing {name} using {num_samples} samples")
            with torch.no_grad(), align_module_device(
                module
            ), self._maybe_onload_hessian(module), CompressionLogger(
                module
            ) as comp_logger:
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
