import tqdm
from compressed_tensors.quantization import disable_quantization, enable_quantization

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    update_weight_zp_scale,
)
from llmcompressor.modifiers.quantization.quantization.mixin import QuantizationMixin

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

        QuantizationMixin.attach_scheme_and_observers(self, state.model)
        state.model.apply(disable_quantization)  # disable quantization until start

        # FUTURE: modify oneshot lifecycle to trigger on_start for on initialize
        if self.calculate_start() == -1:  # one shot
            self.on_start(state)

        return True

    def on_start(self, state: State):
        """
        Begin calibrating activations and weights. Calibrate weights only once on start
        """
        QuantizationMixin.register_calibration_hooks(self, state.model)
        state.model.apply(apply_calibration_status)
        state.model.apply(enable_quantization)

        modules = list(state.model.modules())
        for module in tqdm.tqdm(modules, desc="Calibrating weights"):
            update_weight_zp_scale(module)

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            # TODO: modify lifecycle to end on calibration epoch end
            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True  # TODO: move to super call
        state.model.apply(freeze_module_quantization)  # remove observers
        self.remove_hooks()  # remove hooks

    def on_finalize(self, state: State, **kwargs) -> bool:
        # TODO: modify lifecycle to end on finalize
        if not self.ended_:
            self.on_end(state, None)
