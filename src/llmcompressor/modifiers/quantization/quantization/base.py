from typing import Any, Dict, List, Optional, Union

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
    freeze_module_quantization,
    is_preset_scheme,
    preset_name_to_scheme,
    set_module_for_calibration,
)
from compressed_tensors.quantization.observers.helpers import get_observer_token_count
from loguru import logger
from pydantic import Field
from torch.nn import Module

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.utils.pytorch_helpers import (
    is_moe_model,
    run_calibration_forward,
)

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
    :param targets: list of layer names to quantize if a scheme is provided
    :param disable_quantization_observer_epoch: Epoch to disable updates to the module
        quantization observers. At this point, quantized weights and zero points will
        not be updated. Leave None to not disable observers during QAT. Default is None
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    """

    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    ignore: List[str] = Field(default_factory=list)
    targets: Union[str, List[str], None] = None
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    kv_cache_scheme: Optional[QuantizationArgs] = None
    disable_quantization_observer_epoch: Optional[float] = None
    num_calibration_steps: Optional[int] = None

    calibration_dataloader_: Any = None
    calibration_function_: Any = None

    def on_initialize_structure(self, state: State, **kwargs):
        pass

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

        if self.calculate_start() == -1:  # one-shot
            self._check_calibration_data(config)
            module.apply(set_module_for_calibration)
            self._calibrate_if_possible(module)
            self._check_token_distribution(
                module, threshold=kwargs.get("min_tokens_per_module")
            )
            module.apply(freeze_module_quantization)

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        module = state.model
        module.apply(set_module_for_calibration)

    def on_update(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.BATCH_START:
            if self.check_should_disable_observer(event):
                module = state.model
                module.apply(freeze_module_quantization)

    def on_end(self, state: State, event: Event, **kwargs):
        module = state.model
        module.apply(freeze_module_quantization)

    def on_event(self, state: State, event: Event, **kwargs):
        pass

    def create_init_config(self) -> QuantizationConfig:
        if self.targets is not None and isinstance(self.targets, str):
            self.targets = [self.targets]

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
            default_quant_scheme = QuantizationScheme.default_scheme(
                targets=self.targets
            )
            self.config_groups = {"group_0": default_quant_scheme}
            logger.info(
                "No config groups were provided, generating "
                f"QuantizationScheme.default_scheme = {self.config_groups}"
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
        apply_quantization_config(model, modifier_as_config)
        return modifier_as_config

    def _calibrate_if_possible(self, module: Module):
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

        self._calibrate(module)

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
