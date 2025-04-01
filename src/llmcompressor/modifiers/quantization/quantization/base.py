from typing import Any, Dict, List, Optional, Union

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    apply_quantization_config,
    is_attention_module,
)
from pydantic import Field, field_validator
from torch.nn import Module

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    initialize_quantized_kv_cache,
    register_calibration_hooks,
    remove_quantized_kv_cache,
    update_weight_zp_scale,
)
from llmcompressor.utils import resolve_modifier_quantization_config

__all__ = ["QuantizationModifier"]


class QuantizationModifier(Modifier):
    """
    Enables post training quantization (PTQ) and quantization aware training (QAT) for a
    given module or its submodules. After calibration (PTQ) or the start epoch (QAT),
    the specified module(s) forward pass will emulate quantized execution and the
    modifier will be enabled until training is completed.

    :param config_groups: dictionary specifying quantization schemes to apply to target
        modules. Modules not matching a scheme target will NOT be quantized.
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
    :param targets: list of layer names to quantize if a scheme is provided. Defaults
        to Linear layers
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    """

    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    ignore: List[str] = Field(default_factory=list)
    targets: Union[str, List[str]] = Field(default_factory=lambda: ["Linear"])
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    kv_cache_scheme: Optional[QuantizationArgs] = None
    num_calibration_steps: Optional[int] = None

    @field_validator("targets", mode="before")
    def validate_targets(cls, value: Union[str, List[str]]) -> List[str]:
        if isinstance(value, str):
            return [value]

        return value

    @field_validator("end", mode="before")
    def validate_end(cls, value: Optional[int]) -> Optional[int]:
        if value not in (None, -1):
            raise ValueError(
                "end_epoch is disabled for QuantizationModifier and can only be set to"
                " -1 or None. Given {}".format(value)
            )

        return value

    def on_initialize(self, state: State) -> bool:
        # apply config to model
        config = resolve_modifier_quantization_config(self)
        apply_quantization_config(state.model, config)

        # add observers as modules
        state.model.apply(lambda mod: initialize_observer(mod, base_name="input"))
        state.model.apply(lambda mod: initialize_observer(mod, base_name="weight"))
        state.model.apply(lambda mod: initialize_observer(mod, base_name="output"))
        state.model.apply(initialize_quantized_kv_cache)

        # register hooks to use observers
        state.model.apply(lambda mod: register_calibration_hooks(self, mod))

        return True

    def on_start(self, state: State):
        super().on_start(state)
        state.model.apply(apply_calibration_status)

        # do an initial calibration of the weights
        # TODO: shouldn't this also be done whenever weights are updated?
        state.model.apply(update_weight_zp_scale)

    def on_end(self, state: State, event: Event, **kwargs):
        super().on_end(state, event)
        self.remove_hooks()  # disable observer calibration
        state.model.apply(remove_quantized_kv_cache)  # equivalent to disable kv quant
        state.model.apply(freeze_module_quantization)

    def on_finalize(self, state: State):
        super().on_finalize(state)
