from typing import Any, Dict, List, Optional, Union

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
from loguru import logger
from pydantic import Field, field_validator
from torch.nn import Module

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    calibrate_input_hook,
    calibrate_kv_cache_input_hook,
    calibrate_kv_cache_output_hook,
    calibrate_output_hook,
    freeze_module_quantization,
    initialize_observer,
    set_unset_kv_cache,
    update_weight_zp_scale,
)
from llmcompressor.modifiers.utils.pytorch_helpers import (
    is_moe_model,
    run_calibration_forward,
)
from llmcompressor.observers.helpers import get_observer_token_count

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
    :param disable_quantization_observer_epoch: Epoch to disable updates to the module
        quantization observers. At this point, quantized weights and zero points will
        not be updated. Leave None to not disable observers during QAT. Default is None
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    """

    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    ignore: List[str] = Field(default_factory=list)
    targets: Union[str, List[str]] = Field(default_factory=lambda: ["Linear"])
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    kv_cache_scheme: Optional[QuantizationArgs] = None
    disable_quantization_observer_epoch: Optional[float] = None
    num_calibration_steps: Optional[int] = None

    calibration_dataloader_: Any = None
    calibration_function_: Any = None

    @field_validator("targets", mode="before")
    def validate_targets(cls, value: Union[str, List[str]]) -> List[str]:
        if isinstance(value, str):
            return [value]

        return value

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.end and self.end != -1:
            raise ValueError(
                "end_epoch is disabled for QuantizationModifier and can only be set to"
                " -1 or None. Given {}".format(self.end)
            )

        self.calibration_dataloader_ = state.data.calib
        module = state.model

        # initialize quantization in appropriate modules
        config = self._apply_modifier_to_model(module)
        module.apply(lambda module: initialize_observer(module, base_name="weight"))

        if self.calculate_start() == -1:  # one-shot
            self._check_calibration_data(config)
            module.apply(update_weight_zp_scale)
            module.apply(apply_calibration_status)
            self._calibrate_if_possible(module)
            self._check_token_distribution(
                module, threshold=kwargs.get("min_tokens_per_module")
            )
            module.apply(freeze_module_quantization)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        module = state.model
        module.apply(update_weight_zp_scale)

    def on_update(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.BATCH_START:
            if self.check_should_disable_observer(event):
                module = state.model
                module.apply(freeze_module_quantization)

    def on_end(self, state: State, event: Event, **kwargs):
        module = state.model
        module.apply(freeze_module_quantization)

    def create_init_config(self) -> QuantizationConfig:
        if self.scheme is not None:
            # takes precedence over config_groups

            if isinstance(self.scheme, str) and is_preset_scheme(self.scheme):
                # attach targets to scheme
                self.scheme = {self.scheme: self.targets}

            self.config_groups = {}
            for idx, key in enumerate(self.scheme.keys()):
                if is_preset_scheme(key):
                    scheme = preset_name_to_scheme(key, self.scheme[key])
                else:
                    scheme = QuantizationScheme.model_validate(
                        {"targets": self.scheme[key], **self.scheme}
                    )

                group_name = f"group_{idx}"
                self.config_groups[group_name] = scheme

        if self.config_groups is None or len(self.config_groups) == 0:
            default_quant_scheme = QuantizationScheme(targets=self.targets)
            self.config_groups = {"group_0": default_quant_scheme}
            logger.info(
                f"No config groups were provided, using default {self.config_groups}"
            )

        return QuantizationConfig(
            config_groups=self.config_groups,
            kv_cache_scheme=self.kv_cache_scheme,
            quantization_status=QuantizationStatus.INITIALIZED,
            ignore=self.ignore,
        )

    def calculate_disable_observer_epoch(self) -> float:
        """
        Get the epoch at which we want to disable to quantization observer
        :return epoch to disable at, or -1 if it is not set
        """
        return (
            self.disable_quantization_observer_epoch
            if self.disable_quantization_observer_epoch is not None
            else -1
        )

    def check_should_disable_observer(self, event: Event) -> bool:
        """
        Given the current index, determine if we should disable the observer

        :param event: Event to get index from
        :return: True if observer should be disabled, False otherwise
        """
        disable_epoch = self.calculate_disable_observer_epoch()
        if disable_epoch == -1:
            return False
        if event.current_index >= disable_epoch:
            return True
        return False

    def _check_calibration_data(self, config: QuantizationConfig):
        has_calibration_data = self.calibration_dataloader_ is not None
        requires_calibration = config.requires_calibration_data()
        if self.calculate_start() == -1:  # one shot
            if requires_calibration and not has_calibration_data:
                raise ValueError(
                    "The provided quantization configuration requires calibration data "
                    "but none was provided. Calibration data is required for static "
                    "quantization of input or output activations."
                )
            if not requires_calibration and has_calibration_data:
                logger.info(
                    "Skipping QuantizationModifier calibration, it is not required for "
                    "the provided quantization config."
                )
                self.calibration_dataloader_ = None

    def _apply_modifier_to_model(self, model: Module):
        modifier_as_config = self.create_init_config()
        # Add step to attach kv_cache to the model, if present within the config
        # apply_quantization_config(model, modifier_as_config)
        model.apply(set_unset_kv_cache)
        return modifier_as_config

    def _calibrate_if_possible(self, module: Module):
        # TODO: @dsikka restructure such that all of calibration isn't happening
        # on init
        # flake8: noqa
        """# noqa: E501
        Run calibration if running input/output activation quantization or kv_cache
        quantization.

        Calibration Lifecycle for a single torch.nn.Module:

        initialize_observer():
            if input/output activation:
                - observer = Observer.load_from_registry(...)
                - module.register_module(f"{base_name}_observer", observer)

        register_calibration_hooks():
            if input activation and not dynamic quant (used to call observers before intput QDQ):
                - pre_hook := calibrate_input_hook
            if output activation and not dynamic quant (used to call observers before output QDQ):
                - post_hook := calibrate_kv_cache_output_hook
            if kv_cache quantization (used to set kv_cache to QuantizedKVParameterCache and update k_scale/v_scale)
                - pre_hook := calibrate_kv_cache_input_hook
                - post_hook := calibrate_kv_cache_output_hook

        self._calibrate(module) # run forward pass through model using calibration data
        set_unset_kv_cache() # remove kv_cache objects attached to attention layers
        # initially set in _apply_modifier_to_model
        remove calibration hooks in self.calibration_hooks_
        remove observers

        """
        if self.num_calibration_steps == 0 and self.calibration_dataloader_:
            logger.warning(
                f"num_calibration_steps is {self.num_calibration_steps}."
                f"Calibration data loader will not be used."
            )
        elif self.num_calibration_steps and not self.calibration_dataloader_:
            raise ValueError(
                f"num_calibration_steps is {self.num_calibration_steps}. "
                "Calibration data loader is not set. Pass a "
                "calibration_data_loader with initialize(...) method."
            )

        elif not self.calibration_dataloader_:
            return

        module.apply(lambda model: initialize_observer(model, base_name="input"))
        module.apply(lambda model: initialize_observer(model, base_name="output"))
        module.apply(self.register_calibration_hooks)
        self._calibrate(module)
        module.apply(set_unset_kv_cache)
        self.remove_hooks()

    def register_calibration_hooks(self, module: Module):
        """
        Register hooks for input/output activation or kv_cache quantization.
        """
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            return

        is_attention_module_ = is_attention_module(module)
        input_quant = quantization_scheme.input_activations
        output_quant = quantization_scheme.output_activations

        calibrate_inputs = (
            input_quant and not is_attention_module_ and not input_quant.dynamic
        )

        # Calibrate inputs if an input_quant is provided and not running dynamic quant
        if calibrate_inputs:
            self.register_hook(module, calibrate_input_hook, "forward_pre")

        if output_quant:
            # hooks for attn modules if running kv_cache quant
            if is_attention_module_:
                self.register_hook(
                    module,
                    calibrate_kv_cache_input_hook,
                    "forward_pre",
                    with_kwargs=True,
                )

                self.register_hook(module, calibrate_kv_cache_output_hook, "forward")

            # hooks for output quant if not running dynamic quant
            elif not output_quant.dynamic:
                self.register_hook(module, calibrate_output_hook, "forward")

    def _calibrate(self, module: Module):
        class_name = self.__class__.__name__.replace("PyTorch", "")
        logger.info(
            f"Running {class_name} calibration with "
            f"{len(self.calibration_dataloader_)} samples..."
        )

        module_training = module.training
        module.eval()

        run_calibration_forward(
            module,
            self.calibration_dataloader_,
            self.num_calibration_steps,
            self.calibration_function_,
        )

        if module_training:
            module.train()

    def _check_token_distribution(
        self, model: Module, threshold: Optional[float] = None
    ):
        """
        A helper function that warns when a module has seen
        fewer than threshold % of all the tokens throughout
        the calibration process.
        Checks are only triggered if threshold is not None.
        :param model: the model to validate
        :param threshold: the minimum percentage of tokens
            (out of all the tokens in a batch) a module should
            receive during calibration
        """

        if self.calibration_dataloader_ is None:
            logger.debug("Skipping token distribution check. No calibration data.")
            return

        if not is_moe_model(model):
            logger.debug("Skipping token distribution check. Not a MoE model.")
            return

        if threshold is None:
            logger.warning(
                "Mixture of Experts model detected, but threshold not set. "
                "Defaulting token threshold to 1/num_experts."
            )

            if not hasattr(model.config, "num_local_experts"):
                logger.warning(
                    "Mixture of Experts model detected but `num_local_experts` "
                    "not found in model config. Skipping distribution check."
                )
                return

            threshold = 1 / model.config.num_local_experts
            logger.debug(f"Setting token threshold to {threshold}.")

        all_tokens = self.calibration_dataloader_.dataset["input_ids"]
        total_token_count = sum(len(sample) for sample in all_tokens)
        counter = get_observer_token_count(model)
        for module_name, token_count in counter.items():
            if token_count is None:
                # the module has not been observed
                # or its token_count is not being recorded
                # by the observer (refer to the observer's
                # implementation in the source code)
                continue
            if token_count / total_token_count < threshold:
                logger.warning(
                    f"The module_name: {module_name} "
                    f"received less than {int(threshold * 100)}% "
                    "of calibration batch tokens "
                    f"({token_count}/{total_token_count} tokens). "
                    "This could result may harm the quantization quality."
                )
