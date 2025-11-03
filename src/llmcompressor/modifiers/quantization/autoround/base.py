from typing import Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
    enable_quantization,
)
from compressed_tensors.utils import (
    align_module_device,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import apply_calibration_status
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin

__all__ = ["AutoRoundModifier"]


def normalize_input(cur_inputs):
    # TODO: move it to auto-round
    input_ids = []
    input_others = {}
    positional_inputs = []
    attention_mask = None
    position_ids = None
    cache_position = None
    position_embeddings = (None, None)
    for cur_inp in cur_inputs:
        input_ids.append(cur_inp[0][0][0])
        for key, val in cur_inp[0][1].items():
            if key == "position_ids":
                position_ids = val
            elif key == "position_embeddings":
                position_embeddings = val
            elif key == "cache_position":
                cache_position = val
    input_others["position_ids"] = position_ids
    input_others["positional_inputs"] = positional_inputs
    input_others["attention_mask"] = attention_mask
    input_others["position_embeddings"] = position_embeddings
    input_others["cache_position"] = cache_position
    return input_ids, input_others


def _is_decoding_layer(module, name):
    return "decoderlayer" in module.__class__.__name__.lower()


class _LLModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()

    def forward(self, *args, **kwargs):
        for layer in self.layers:
            res = layer(*args, **kwargs)
        return res


class _PretrainModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _LLModelWrapper()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _wrap_decoding_layer(layer: torch.nn.Module) -> _PretrainModelWrapper:
    wrapped_model = _PretrainModelWrapper()
    wrapped_model.model.layers.append(layer)
    first_param = next(layer.parameters())
    wrapped_model.dtype = first_param.dtype
    return wrapped_model


class AutoRoundModifier(Modifier, QuantizationMixin):
    """
    Implements the AutoRound algorithm from https://arxiv.org/pdf/2309.05516. This modifier
    leverages signed gradient descent (SignSGD) and block-wise loss to optimize rounding values
    and weight clipping in a few steps.

    | Sample yaml:
    | test_stage:
    |    obcq_modifiers:
    |      AutoRoundModifier:
    |          iters: 200
    |          config_groups:
    |            group_0:
    |                targets:
    |                  - "Linear"
    |                input_activations: null
    |                output_activations: null
    |                weights:
    |                    num_bits: 4
    |                    type: "int"
    |                    symmetric: true
    |                    strategy: group
    |                    group_size: 128

    Lifecycle:
        - on_initialize
            - apply config to model
        - on_start
            - add input/output capture hooks to decoding layers
        - on_sequential_epoch_end
            - quantize_weight
        - on_finalize
            - remove_hooks()
            - model.apply(freeze_module_quantization)

    :param sequential_targets: list of layer names to compress during AutoRound, or
        '__ALL__' to compress every layer in the model

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
    """

    # AutoRound modifier arguments
    sequential_targets: Union[str, List[str], None] = None
    iters: Optional[int] = 200
    # TODO: this does not serialize / will be incorrectly written

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _cur_layer_idx = PrivateAttr(default=0)
    _all_module_input: Dict[str, List[Tuple]] = PrivateAttr(default_factory=dict)
    _all_module_output: Dict[str, List[Tuple]] = PrivateAttr(default_factory=dict)

    def resolve_quantization_config(self) -> QuantizationConfig:
        config = super().resolve_quantization_config()
        return config

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the AutoRound algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # prepare module names
        self._module_names = {
            m: name
            for name, m in match_named_modules(state.model, self.targets, self.ignore)
        }
        # add temporary names to all modules for debugging
        for name, mod in state.model.named_modules():
            mod._tmp_name = name
        # freeze all model parameters
        for name, param in state.model.named_parameters():
            param.requires_grad_(False)
        return True

    def start_calibration(self, model: torch.nn.Module):
        """
        Register activation calibration hooks and enable quantization as we calibrate

        :param model: model to prepare for calibration
        """

        for _, module in match_named_modules(model, self.targets, self.ignore):
            # Note: No need to register observers for auto-round
            # self._initialize_observers(module)
            self._calibration_hooks |= self._initialize_hooks(module)
            apply_calibration_status(module)

        model.apply(enable_quantization)  # quantize at the same time as calibrate

    def input_capture_hook(self, module, *args, **kwargs):
        self._all_module_input[module._tmp_name].append((args, kwargs))

    def output_capture_hook(self, module, *args, **kwargs):
        self._all_module_output[module._tmp_name].append((args, kwargs))

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register quantization calibration hooks
        # assume quantization has been initialized by this modifier or one before it
        self.start_calibration(state.model)
        for name, module in state.model.named_modules():
            if _is_decoding_layer(module, name):
                # register input/output capture hooks for decoding layers
                logger.warning(
                    f">> Registering input/output capture hooks for decoding layer {getattr(module, '_tmp_name', '')} || {name}"
                )
                self.register_hook(
                    module, self.input_capture_hook, "forward_pre", with_kwargs=True
                )
                self.register_hook(
                    module, self.output_capture_hook, "forward", with_kwargs=True
                )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self.apply_autoround(state)
            self.post_autoround_cleanup()

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state, None)

    def apply_autoround(self, state):
        cur_layer_idx = self._cur_layer_idx
        self._cur_layer_idx += 1
        logger.info(f">>||>> AutoRound for decoding layer index {cur_layer_idx}")
        if cur_layer_idx >= len(state.model.model.layers):
            # skip the lm_head layer
            logger.info(
                ">>||>> All decoding layers have been processed for AutoRound."
            )
            # self.compress_modules(return_directly=False)
            return
        decoding_layer = state.model.model.layers[cur_layer_idx]
        logger.debug(
            f">>||>> Strating AutoRound for decoding layer {getattr(decoding_layer, '_tmp_name', '')}"
        )

        wrapped_model = _wrap_decoding_layer(decoding_layer)

        with torch.enable_grad(), align_module_device(decoding_layer):
            import auto_round

            ar = auto_round.AutoRound(
                model=wrapped_model,
                tokenizer="",
                scheme="W4A16",
                iters=self.iters,
                enable_quanted_input=False,
                enable_torch_compile=True,
            )

            ar.configure_layer_config()

            input_name = f"model.layers.{cur_layer_idx}"
            cur_inputs = self._all_module_input[input_name]
            input_ids, input_others = normalize_input(cur_inputs)
            decoding_layer.tuning_device = torch.device("cuda")

            ar.quantize_block(
                block=decoding_layer,
                input_ids=input_ids,
                input_others=input_others,
                q_input=None,
                device="cuda",
            )
            # Update offload parameters and remove temporary attributes
            for name, module in decoding_layer.named_modules():
                if hasattr(module, "weight_scale") and hasattr(
                    module, "weight_zero_point"
                ):
                    logger.debug(
                        f"Updating offload parameters for module {getattr(module, '_tmp_name', '')} || {name}"
                    )
                    # Note: The model's weight is already quantized and dequantized in-place by auto-round.
                    weight_scale = module.scale
                    del module.scale
                    del module.zp
                    # TODO: update weight as well
                    update_offload_parameter(module, "weight_scale", weight_scale)
        decoding_layer.eval()

    def post_autoround_cleanup(self):
        self._all_module_input.clear()
        self._all_module_output.clear()

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True
        QuantizationMixin.end_calibration(self, state.model)
        self.remove_hooks()  # remove gptq hooks

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        if not self.ended_:
            self.on_end(state, None)

        return True
