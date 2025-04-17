from typing import Any, Dict, List, Optional, Union

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
    is_attention_module,
    is_preset_scheme,
    preset_name_to_scheme,
)
from pydantic import Field, field_validator

from llmcompressor.modifiers.quantization.calibration import (
    calibrate_input_hook,
    calibrate_kv_cache_input_hook,
    calibrate_kv_cache_output_hook,
    calibrate_output_hook,
    initialize_observer,
    initialize_quantized_kv_cache,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin

__all__ = ["QuantizationMixin"]


class QuantizationMixin(HooksMixin):
    """
    Mixin which enables a Modifier to act as a quantization config, attching observers,
    calibration hooks, and compression wrappers to modifiers

    Lifecycle:
        - QuantizationMixin.attach_scheme_and_observers(model)
            - Wraps model forward and attaches quantization scheme and observers
        - QuantizationMixin.register_calibration_hooks(model)
            - Registers calibration hooks which utilize observers to calibrate qparams
        - model.apply(apply_calibration_status)
        - [ Calibrate model ]
        - model.apply(freeze_module_quantization)
            - Remove observers
        - self.remove_hooks()
            - Remove calibration hooks

        Scheme is left attached to modules after PTQ finishes

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

    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    targets: Union[str, List[str]] = Field(default_factory=lambda: ["Linear"])
    ignore: List[str] = Field(default_factory=list)
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    kv_cache_scheme: Optional[QuantizationArgs] = None

    @field_validator("targets", mode="before")
    def validate_targets(cls, value: Union[str, List[str]]) -> List[str]:
        if isinstance(value, str):
            return [value]

        return value

    @field_validator("scheme", mode="before")
    def validate_scheme(
        cls, value: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Union[str, Dict[str, Any]]]:
        if isinstance(value, str) and not is_preset_scheme(value):
            raise ValueError(
                "`scheme` must either be a preset scheme name or a dictionary "
                "of preset scheme names"
            )

        if isinstance(value, dict):
            for scheme_name in value.keys():
                cls.validate_scheme(scheme_name)

            for key, target in value.items():
                value[key] = cls.validate_targets(target)

        return value

    def attach_scheme_and_observers(self, model: torch.nn.Module):
        """
        Apply this modifier as a quantization config to the model. Attach observers
        according to the schemes attached to each module
        """
        config = self.resolve_quantization_config()
        apply_quantization_config(model, config)
        model.apply(self._initialize_observers)

    def register_calibration_hooks(self, model: torch.nn.Module):
        """
        Register activation calibration hooks (including kv_cache quantization)
        """
        model.apply(self._initialize_hooks)

    def has_config(self) -> bool:
        return not (
            self.config_groups is None
            and self.targets == ["Linear"]
            and self.ignore == []
            and self.scheme is None
            and self.kv_cache_scheme is None
        )

    def resolve_quantization_config(self) -> QuantizationConfig:
        """
        Returns the quantization config specified by this modifier
        """
        scheme = self.scheme
        targets = self.targets
        config_groups = self.config_groups
        kv_cache_scheme = self.kv_cache_scheme
        ignore = self.ignore

        if scheme is not None and config_groups is not None:
            raise ValueError("Please specify either `scheme` or `config_groups`")

        if scheme is not None:
            # takes precedence over config_groups

            if isinstance(scheme, str) and is_preset_scheme(scheme):
                # attach targets to scheme
                scheme = {scheme: targets}

            config_groups = {}
            for idx, key in enumerate(scheme.keys()):
                if is_preset_scheme(key):
                    scheme = preset_name_to_scheme(key, scheme[key])
                else:
                    scheme = QuantizationScheme.model_validate(
                        {"targets": scheme[key], **scheme}
                    )

                group_name = f"group_{idx}"
                config_groups[group_name] = scheme

        if config_groups is None or len(config_groups) == 0:
            default_quant_scheme = QuantizationScheme(targets=targets)
            config_groups = {"group_0": default_quant_scheme}

        return QuantizationConfig(
            config_groups=config_groups,
            kv_cache_scheme=kv_cache_scheme,
            quantization_status=QuantizationStatus.INITIALIZED,
            ignore=ignore,
        )

    def _initialize_observers(self, module: torch.nn.Module):
        if not hasattr(module, "quantization_scheme"):
            return

        scheme: QuantizationScheme = module.quantization_scheme
        input = scheme.input_activations and not scheme.input_activations.dynamic
        weight = scheme.weights is not None
        output = scheme.output_activations and not scheme.output_activations.dynamic
        is_attention = is_attention_module(module)

        # input activations
        if input:
            initialize_observer(module, base_name="input")

        # weight observers (used by `update_weight_zp_scale` or child modifier)
        if weight:
            initialize_observer(module, base_name="weight")

        # kv_cache activations. Within `apply_quantization_config`, the config is
        # modified to use attention output quantization if a kv_cache_scheme exists
        if is_attention and output:
            initialize_quantized_kv_cache(module)

        # output activations
        elif output:
            initialize_observer(module, base_name="output")

    def _initialize_hooks(self, module: torch.nn.Module):
        if not hasattr(module, "quantization_scheme"):
            return

        scheme: QuantizationScheme = module.quantization_scheme
        input = scheme.input_activations and not scheme.input_activations.dynamic
        output = scheme.output_activations and not scheme.output_activations.dynamic
        is_attention = is_attention_module(module)

        # input activations
        if input:
            self.register_hook(module, calibrate_input_hook, "forward_pre")

        # kv_cache activations. Within `apply_quantization_config`, the config is
        # modified to use attention output quantization if a kv_cache_scheme exists
        if is_attention and output:
            self.register_hook(
                module, calibrate_kv_cache_input_hook, "forward_pre", with_kwargs=True
            )
            self.register_hook(module, calibrate_kv_cache_output_hook, "forward")

        # output activations
        elif output:
            self.register_hook(module, calibrate_output_hook, "forward")
