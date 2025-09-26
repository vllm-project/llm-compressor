import tqdm
from compressed_tensors.utils import match_named_modules

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import (
    update_weight_global_scale,
    update_weight_zp_scale,
)
from llmcompressor.modifiers.quantization.quantization.mixin import QuantizationMixin
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales

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
                "QuantizationModifier requires that quantization fields be specified"
            )
        QuantizationMixin.initialize_quantization(self, state.model)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        """
        Begin calibrating activations and weights. Calibrate weights only once on start
        """
        self.started_ = True
        QuantizationMixin.start_calibration(self, state.model)

        named_modules = list(
            match_named_modules(state.model, self.targets, self.ignore)
        )
        # TODO: this step can be combined with update_weight_zp_scale
        # once update_fused_layer_weight_global_scales is removed
        # and not required by vLLM
        for _, module in tqdm.tqdm(named_modules, desc="Updating global scales"):
            update_weight_global_scale(module)

        # NOTE: update_fused_layer_weight_global_scales operates on Attention
        # and MLP layers, not quantizable Linear layers. Rather than running
        # on targeted modules, we need to run on all modules.
        # Because this call is idempotent, setting all global_scales to the
        # min value, it is ok to run potentially multiple times for all modules
        for module in tqdm.tqdm(state.model.modules(), desc="Fusing global scales"):
            update_fused_layer_weight_global_scales(module)

        for _, module in tqdm.tqdm(named_modules, desc="Calibrating weights"):
            update_weight_zp_scale(module)

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True
        QuantizationMixin.end_calibration(
            self, state.model
        )  # keep quantization enabled

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, None)
