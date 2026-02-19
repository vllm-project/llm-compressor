from typing import Any, Dict, List, Optional, Set, Union

import torch
from compressed_tensors.modeling import (
    IMPL_ATTR,
    KV_CACHE_ATTR,
)
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
from compressed_tensors.utils import match_named_modules
from pydantic import Field, PrivateAttr, field_validator
from torch.utils.hooks import RemovableHandle

from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    calibrate_input_hook,
    calibrate_key_hook,
    calibrate_output_hook,
    calibrate_query_hook,
    calibrate_value_hook,
    freeze_module_quantization,
    initialize_observer,
    reset_quantization_status,
)
from llmcompressor.modifiers.quantization.group_size_validation import (
    validate_group_size_divisibility,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.utils import targets_embeddings, untie_word_embeddings

__all__ = ["QuantizationMixin"]


class QuantizationMixin(HooksMixin):
    """
    Mixin which enables a Modifier to act as a quantization config, attaching observers,
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
    :param targets: list of layer names to quantize if a scheme is provided. If unset,
        will contain all targets listed in config_groups. If config_groups is also
        unset, will default to ["Linear"] (i.e. all Linear layers will be targeted).
        This field is not the source of truth for finding all matching target layers
        in a model. Additional information can be stored in `config_groups`. Use
        self.resolved_targets instead.
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
            - quantizes the outputs of the aforementioned layers, so that
              keys and values are compressed before storing them in the cache
        There is an explicit assumption that the model contains modules with
        `k_proj` and `v_proj` in their names. If this is not the case
        and kv_cache_scheme != None, the quantization of kv cache will fail
    :param weight_observer: optional observer name for weight quantization.
        Overrides the default observer specified in the scheme. Valid values
        include "minmax", "mse", "static_minmax", "memoryless_minmax", "memoryless_mse".
    :param input_observer: optional observer name for input activation quantization.
        Overrides the default observer specified in the scheme. Valid values
        include "minmax", "mse", "static_minmax", "memoryless_minmax", "memoryless_mse".
    :param output_observer: optional observer name for output activation quantization.
        Overrides the default observer specified in the scheme. Valid values
        include "minmax", "mse", "static_minmax", "memoryless_minmax", "memoryless_mse".
    :param observer: optional dictionary to specify observers for multiple quantization
        types at once. Keys can be "weights", "input", or "output". Values are observer
        names. Example: {"weights": "MSE", "input": "MSE"}. If both individual
        observer parameters (weight_observer, input_observer, output_observer) and
        observer dict are provided, the observer dict takes precedence.
    :param bypass_divisibility_checks: if True, skip the check that weight columns
        are divisible by group_size for GROUP/TENSOR_GROUP. Use when your runtime
        (e.g. vLLM) supports non-divisible dimensions. Defaults to False.
    """

    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    # NOTE: targets is not the sole source of truth for finding all matching target
    # layers in a model. Additional information can be stored in `config_groups`
    # Use self.resolved_targets as source of truth.
    targets: Union[str, List[str]] = Field(default_factory=lambda: ["Linear"])
    ignore: List[str] = Field(default_factory=list)
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    kv_cache_scheme: Optional[QuantizationArgs] = None
    # Observer parameters for easy specification
    weight_observer: Optional[str] = None
    input_observer: Optional[str] = None
    output_observer: Optional[str] = None
    observer: Optional[Dict[str, str]] = None
    bypass_divisibility_checks: bool = False

    _calibration_hooks: Set[RemovableHandle] = PrivateAttr(default_factory=set)
    _resolved_config: Optional[QuantizationConfig] = PrivateAttr(None)

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

    @field_validator("observer", mode="before")
    def validate_observer(cls, value: Any) -> Optional[Dict[str, str]]:
        """
        Validate observer dictionary format. Accepts keys: 'weights', 'input', 'output'
        """
        if value is None:
            return value

        if not isinstance(value, dict):
            raise ValueError("`observer` must be a dictionary")

        valid_keys = {"weights", "input", "output"}
        for key in value.keys():
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid observer key '{key}'. Valid keys are: {valid_keys}"
                )
            if not isinstance(value[key], str):
                raise ValueError(f"Observer value for '{key}' must be a string")

        return value

    @property
    def resolved_config(self) -> QuantizationConfig:
        """
        Quantization config needs to be resolved just once based on
        scheme and config_groups inputs.
        """
        if self._resolved_config is None:
            self._resolved_config = self.resolve_quantization_config()
        return self._resolved_config

    @property
    def resolved_targets(self) -> Set[str]:
        """
        Set of all resolved targets, i.e. all unique targets listed
        in resolved quantization config.
        Use this property instead of the targets field, as targets can
        also come from config_groups depending on how recipe is configured.
        """
        targets = set()
        for config_group in self.resolved_config.config_groups.values():
            for target in config_group.targets:
                targets.add(target)

        if self.resolved_config.kv_cache_scheme is not None:
            # TODO: decouple reliance on this regex for matching attention
            targets.add("re:.*self_attn$")

        return targets

    def initialize_quantization(self, model: torch.nn.Module):
        """
        Attach quantization schemes to modules in the model according to
        the quantization config specified on this modifier

        :param model: model to attach schemes and observers to
        """

        for _, module in match_named_modules(model, self.resolved_targets, self.ignore):
            reset_quantization_status(module)  # reset any previously applied qconfigs

        apply_quantization_config(model, self.resolved_config)

        if not self.bypass_divisibility_checks:
            validate_group_size_divisibility(model, self.resolved_targets, self.ignore)

        # disable quantization until calibration
        model.apply(disable_quantization)

    def start_calibration(self, model: torch.nn.Module):
        """
        Attach observers, register activation calibration hooks (including
        kv_cache quantization) and enable quantization as we calibrate

        :param model: model to prepare for calibration
        """
        targets = match_named_modules(model, self.resolved_targets, self.ignore)
        if targets_embeddings(model, targets):
            untie_word_embeddings(model)

        for _, module in match_named_modules(model, self.resolved_targets, self.ignore):
            self._initialize_observers(module)
            self._calibration_hooks |= self._initialize_hooks(module)
            apply_calibration_status(module)

        model.apply(enable_quantization)  # quantize at the same time as calibrate

    def end_calibration(self, model: torch.nn.Module):
        """
        Remove calibration hooks and observers, and set the model status to frozen.
        Keep quantization enabled for future operations

        :param model: model to end calibration for
        """
        self.remove_hooks(self._calibration_hooks)
        for _, module in match_named_modules(model, self.resolved_targets, self.ignore):
            freeze_module_quantization(module)  # remove observers

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
                    scheme_obj = preset_name_to_scheme(key, scheme[key])
                else:
                    scheme_obj = QuantizationScheme.model_validate(
                        {"targets": scheme[key], **scheme}
                    )

                # Apply observer overrides if specified
                scheme_obj = self._apply_observer_overrides(scheme_obj)

                group_name = f"group_{idx}"
                config_groups[group_name] = scheme_obj

        if config_groups is None or len(config_groups) == 0:
            default_quant_scheme = QuantizationScheme(targets=targets)
            # Apply observer overrides to default scheme as well
            default_quant_scheme = self._apply_observer_overrides(default_quant_scheme)
            config_groups = {"group_0": default_quant_scheme}
        elif scheme is None:
            # Apply observer overrides to all config groups when config_groups
            # was provided directly (not derived from scheme)
            for scheme_obj in config_groups.values():
                self._apply_observer_overrides(scheme_obj)

        return QuantizationConfig(
            config_groups=config_groups,
            kv_cache_scheme=kv_cache_scheme,
            quantization_status=QuantizationStatus.INITIALIZED,
            ignore=ignore,
        )

    def _apply_observer_overrides(
        self, scheme: QuantizationScheme
    ) -> QuantizationScheme:
        """
        Apply observer overrides from weight_observer, input_observer, output_observer,
        or observer dict to the quantization scheme.

        :param scheme: QuantizationScheme to modify
        :return: Modified QuantizationScheme with observers applied
        """
        # Validate that both individual params and dict are not specified
        has_individual = (
            self.weight_observer is not None
            or self.input_observer is not None
            or self.output_observer is not None
        )
        if has_individual and self.observer is not None:
            raise ValueError(
                "Cannot specify both individual observer parameters (weight_observer, "
                "input_observer, output_observer) and observer dict. "
                "Please use either individual parameters or the observer dict."
            )

        # Resolve observer values from individual params or dict
        weight_obs = self.weight_observer
        input_obs = self.input_observer
        output_obs = self.output_observer

        # Override with dict values if provided
        if self.observer is not None:
            weight_obs = self.observer.get("weights", weight_obs)
            input_obs = self.observer.get("input", input_obs)
            output_obs = self.observer.get("output", output_obs)

        # Apply observers to QuantizationArgs if specified
        update_map = [
            (weight_obs, "weights"),
            (input_obs, "input_activations"),
            (output_obs, "output_activations"),
        ]

        for obs_value, scheme_attr in update_map:
            q_args = getattr(scheme, scheme_attr, None)
            if obs_value is not None and q_args is not None:
                args_dict = q_args.model_dump()
                args_dict["observer"] = obs_value
                setattr(scheme, scheme_attr, QuantizationArgs.model_validate(args_dict))

        return scheme

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
            if not is_attention:
                initialize_observer(module, base_name="input")
            else:
                if hasattr(module, IMPL_ATTR):
                    initialize_observer(module, base_name="q")
                if hasattr(module, KV_CACHE_ATTR):
                    initialize_observer(module, base_name="k")
                    initialize_observer(module, base_name="v")

        # weight observers (used by `update_weight_zp_scale` or child modifier)
        if weight:
            initialize_observer(module, base_name="weight")

        # output activations
        if output:
            initialize_observer(module, base_name="output")

    def _initialize_hooks(self, module: torch.nn.Module) -> Set[RemovableHandle]:
        hooks = set()
        if not hasattr(module, "quantization_scheme"):
            return hooks

        scheme: QuantizationScheme = module.quantization_scheme
        input = scheme.input_activations and scheme.input_activations.dynamic in (
            False,
            DynamicType.LOCAL,
        )
        output = scheme.output_activations and not scheme.output_activations.dynamic
        is_attention = is_attention_module(module)

        # input activations
        if input:
            if not is_attention:
                hooks.add(
                    self.register_hook(module, calibrate_input_hook, "forward_pre")
                )
            else:
                if hasattr(module, IMPL_ATTR):
                    hooks.add(self.register_hook(module, calibrate_query_hook, "query"))
                if hasattr(module, KV_CACHE_ATTR):
                    hooks.add(self.register_hook(module, calibrate_key_hook, "key"))
                    hooks.add(self.register_hook(module, calibrate_value_hook, "value"))

        # output activations
        if output:
            hooks.add(self.register_hook(module, calibrate_output_hook, "forward"))

        return hooks
