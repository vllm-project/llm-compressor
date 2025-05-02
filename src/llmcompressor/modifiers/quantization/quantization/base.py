import torch
import tqdm
from loguru import logger

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import update_weight_zp_scale
from llmcompressor.modifiers.quantization.quantization.mixin import QuantizationMixin
from llmcompressor.modifiers.utils.pytorch_helpers import run_calibration_forward
from llmcompressor.utils.helpers import calibration_forward_context

__all__ = ["QuantizationModifier"]


class QuantizationModifier(Modifier, QuantizationMixin):
    """
    Enables post training quantization (PTQ) and quantization aware training (QAT) for a
    given module or its submodules. After calibration (PTQ) or the start epoch (QAT),
    the specified module(s) forward pass will emulate quantized execution and the
    modifier will be enabled until training is completed.

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

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Prepare to calibrate activations and weights

        According to the quantization config, a quantization scheme is attached to each
        targeted module. The module's forward call is also overwritten to perform
        quantization to inputs, weights, and outputs.

        Then, according to the module's quantization scheme, observers and calibration
        hooks are added. These hooks are disabled until the modifier starts.
        """
        if not QuantizationMixin.has_config(self):
            raise ValueError(
                "QuantizationModifier requires that quantization fields to be specified"
            )

        QuantizationMixin.initialize_quantization(self, state.model)

        # FUTURE: modify oneshot lifecycle to trigger on_start for on initialize
        if self.calculate_start() == -1:  # one shot
            self.on_start(state)

        return True

    def on_start(self, state: State):
        """
        Begin calibrating activations and weights. Calibrate weights only once on start
        """
        QuantizationMixin.start_calibration(self, state.model)

        modules = list(state.model.modules())
        for module in tqdm.tqdm(modules, desc="Calibrating weights"):
            update_weight_zp_scale(module)

        # FUTURE: below will be removed after pipeline extraction
        if self.calculate_start() == -1:  # one shot
            self._calibrate_if_possible(state)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        QuantizationMixin.end_calibration(
            self, state.model
        )  # keep quantization enabled

    def on_finalize(self, state: State, **kwargs) -> bool:
        # TODO: modify lifecycle so modifiers end on finalize
        if not self.ended_:
            self.on_end(state, None)

    def _calibrate_if_possible(self, state: State):
        model = state.model
        calibration_dataloader = state.data.calib
        config = QuantizationMixin.resolve_quantization_config(self)

        has_calibration_data = calibration_dataloader is not None
        requires_calibration = config.requires_calibration_data()
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
            return

        if not requires_calibration:
            return

        self._calibrate(model, calibration_dataloader)

    def _calibrate(self, module: torch.nn.Module, data: torch.utils.data.DataLoader):
        class_name = self.__class__.__name__.replace("PyTorch", "")
        logger.info(f"Running {class_name} calibration with {len(data)} samples...")

        with calibration_forward_context(module):
            run_calibration_forward(module, data)
