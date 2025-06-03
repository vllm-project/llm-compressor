from typing import Any, Dict, List, Optional, Set, Union

import torch
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
    disable_quantization,
    enable_quantization,
    is_attention_module,
    is_preset_scheme,
    preset_name_to_scheme,
)
from pydantic import Field, PrivateAttr, field_validator
from torch.utils.hooks import RemovableHandle

from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    calibrate_input_hook,
    calibrate_kv_cache_input_hook,
    calibrate_kv_cache_output_hook,
    calibrate_output_hook,
    freeze_module_quantization,
    initialize_observer,
    initialize_quantized_kv_cache,
    reset_quantization_status,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin

__all__ = ["QuantizationMixin"]


class QuantizationMixin(HooksMixin):
    """
    Mixin which enables a Modifier to act as a quantization config, attching observers,
    calibration hooks, and compression wrappers to modifiers

    Lifecycle:
        - on_initialize: QuantizationMixin.initialize_quantization
            - Attach schemes to modules
            - Attach observers to modules
            - Disable quantization until calibration starts/finishes
        - on_start: QuantizationMixin.start_calibration
            - Attach calibration hooks
            - Apply calibration status
            - Enable quantization during calibration
        - on_end: QuantizationMixin.end_calibration
            - Remove calibration hooks
            - Apply freeze status
            - Keep quantization enabled for future steps
        NOTE: QuantizationMixin does not update scales and zero-points on its own,
          as this is not desired for all Modifiers inheriting from it. Modifier must
          explicitly call `update_weight_zp_scale`.
          See QuantizationModifier.on_start method for example

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

    _calibration_hooks: Set[RemovableHandle] = PrivateAttr(default_factory=set)

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

    def initialize_quantization(self, model: torch.nn.Module):
        """
        Attach quantization schemes and observers to modules in the model according to
        the quantization config specified on this modifier

        :param model: model to attach schemes and observers to
        """
        reset_quantization_status(model)  # reset any previously applied qconfigs

        # apply scheme and status to model
        config = self.resolve_quantization_config()
        apply_quantization_config(model, config)

        # apply observers, disable quantization until calibration
        model.apply(self._initialize_observers)
        model.apply(disable_quantization)

    def start_calibration(self, model: torch.nn.Module):
        """
        Register activation calibration hooks (including kv_cache quantization) and
        enable quantization as we calibrate

        :param model: model to prepare for calibration
        """
        self._calibration_hooks = self._initialize_hooks(model)
        model.apply(apply_calibration_status)
        model.apply(enable_quantization)  # quantize at the same time as calibrate

    def end_calibration(self, model: torch.nn.Module):
        """
        Remove calibration hooks and set the model status to frozen. Keep quantization
        enabled for future operations

        :param model: model to end calibration for
        """
        self.remove_hooks(self._calibration_hooks)
        model.apply(freeze_module_quantization)  # remove observers
        model.apply(enable_quantization)  # keep quantization enabled

    def has_config(self) -> bool:
        """
        Determine if the user has specified a quantization config on this modifier
        """
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
        input = scheme.input_activations and scheme.input_activations.dynamic in (
            False,
            DynamicType.LOCAL,
        )
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

    def _initialize_hooks(self, model: torch.nn.Module) -> Set[RemovableHandle]:
        hooks = set()
        for module in model.modules():
            if not hasattr(module, "quantization_scheme"):
                continue

            scheme: QuantizationScheme = module.quantization_scheme
            input = scheme.input_activations and scheme.input_activations.dynamic in (
                False,
                DynamicType.LOCAL,
            )
            output = scheme.output_activations and not scheme.output_activations.dynamic
            is_attention = is_attention_module(module)

            # input activations
            if input:
                hooks.add(
                    self.register_hook(module, calibrate_input_hook, "forward_pre")
                )

            # kv_cache activations. Within `apply_quantization_config`, the config is
            # modified to use attention output quantization if a kv_cache_scheme exists
            if is_attention and output:
                hooks.add(
                    self.register_hook(
                        module,
                        calibrate_kv_cache_input_hook,
                        "forward_pre",
                        with_kwargs=True,
                    )
                )
                hooks.add(
                    self.register_hook(
                        module, calibrate_kv_cache_output_hook, "forward"
                    )
                )

            # output activations
            elif output:
                hooks.add(self.register_hook(module, calibrate_output_hook, "forward"))

        return hooks
